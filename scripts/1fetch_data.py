"""
fetch_data.py
=============
Étape 1 du projet : collecte des données ATP Tour depuis l'API TennisMyLife.

Ce script :
1. Interroge l'API `api/data-files` pour obtenir la liste exhaustive des CSV disponibles.
2. Filtre et télécharge les CSV ATP Tour (fichier consolidé + tournois en cours)
   dans data/raw/, sans jamais modifier les fichiers une fois téléchargés.
3. Vérifie l'intégrité de chaque fichier téléchargé (colonnes attendues, nombre de lignes > 0).
4. Génère un rapport de valeurs manquantes par colonne et par année dans outputs/.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --years 2022 2023 2024 2025 2026
    python scripts/fetch_data.py --include-challenger

Toute exécution est idempotente : relancer le script re-télécharge et écrase
les fichiers dans 'data/raw/'
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "collection"

API_DATA_FILES = "https://stats.tennismylife.org/api/data-files"
BASE_DATA_URL = "https://stats.tennismylife.org/data"

DEFAULT_YEARS = list(range(2000, 2027))  

# Colonnes minimales attendues dans un CSV de matchs ATP Tour.
EXPECTED_COLUMNS = {
    "tourney_id",
    "tourney_name",
    "surface",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "winner_rank",
    "loser_id",
    "loser_name",
    "loser_rank",
    "score",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
}

REQUEST_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# Structures de résultat
# ---------------------------------------------------------------------------

@dataclass
class FileCheckResult:
    filename: str
    path: Path
    ok: bool
    n_rows: int = 0
    n_cols: int = 0
    missing_expected_columns: set[str] = field(default_factory=set)
    error: str | None = None
    missing_pct_by_column: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Réseau
# ---------------------------------------------------------------------------

def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "github"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        return resp.read()


def fetch_file_list() -> list[dict]:
    """Interroge l'API data-files et renvoie la liste brute des fichiers."""
    print(f"→ Interrogation de l'API : {API_DATA_FILES}")
    try:
        raw = _http_get(API_DATA_FILES)
        payload = json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"  ⚠ Impossible d'interroger l'API ({exc}). "
              f"Bascule sur construction d'URLs directes.")
        return []
    files = payload.get("files", [])
    print(f"  {len(files)} fichiers listés par l'API.")
    return files


def build_target_urls(years: list[int], include_challenger: bool, include_quali: bool) -> dict[str, str]:
    """
    Construit le dictionnaire {nom_fichier: url} des cibles ATP Tour à télécharger.
    Utilisé en secours si l'API data-files est indisponible, et pour filtrer
    la liste retournée par l'API sur ce qui nous intéresse (Étape 1 = ATP Tour).
    """
    # NB: ATP_Database.csv est volontairement exclu par défaut : vérification
    # empirique (30/08/2026) montre qu'il expose un schéma consolidé à 12
    # colonnes, incompatible avec le format match-level à 50 colonnes des
    # fichiers annuels. Les CSV annuels + ongoing_tourneys.csv suffisent.
    targets: dict[str, str] = {
        "ongoing_tourneys.csv": f"{BASE_DATA_URL}/ongoing_tourneys.csv",
    }
    for year in years:
        targets[f"{year}.csv"] = f"{BASE_DATA_URL}/{year}.csv"

    if include_challenger:
        targets["challenger_ongoing_tourneys.csv"] = f"{BASE_DATA_URL}/challenger_ongoing_tourneys.csv"
        for year in years:
            targets[f"{year}_challenger.csv"] = f"{BASE_DATA_URL}/{year}_challenger.csv"

    if include_quali:
        for year in years:
            targets[f"{year}_atp_quali.csv"] = f"{BASE_DATA_URL}/atp_quali/{year}_atp_quali.csv"

    return targets


def download_files(targets: dict[str, str]) -> list[tuple[str, bool, str | None]]:
    """Télécharge chaque fichier cible dans data/raw/. Retourne un journal (nom, succès, erreur)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log: list[tuple[str, bool, str | None]] = []
    for name, url in targets.items():
        dest = RAW_DIR / name
        try:
            data = _http_get(url)
            if len(data) == 0:
                raise ValueError("fichier vide reçu")
            dest.write_bytes(data)
            print(f"  ✓ {name} ({len(data) / 1024:.1f} Ko)")
            log.append((name, True, None))
        except Exception as exc:  # noqa: BLE001 - on veut logguer et continuer
            print(f"  ✗ {name} — échec : {exc}")
            log.append((name, False, str(exc)))
    return log


# ---------------------------------------------------------------------------
# Vérification d'intégrité
# ---------------------------------------------------------------------------

def check_file(path: Path) -> FileCheckResult:
    """Vérifie qu'un CSV téléchargé a des lignes et les colonnes attendues,
    et calcule le taux de valeurs manquantes par colonne."""
    if not path.exists():
        return FileCheckResult(filename=path.name, path=path, ok=False, error="fichier absent")

    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:  # noqa: BLE001
        return FileCheckResult(filename=path.name, path=path, ok=False, error=f"lecture impossible: {exc}")

    n_rows = len(rows)
    n_cols = len(fieldnames)
    missing_expected = EXPECTED_COLUMNS - fieldnames

    missing_pct: dict[str, float] = {}
    if n_rows > 0:
        for col in fieldnames:
            n_missing = sum(1 for r in rows if not r.get(col))
            missing_pct[col] = round(100 * n_missing / n_rows, 2)

    ok = n_rows > 0 and len(missing_expected) == 0

    return FileCheckResult(
        filename=path.name,
        path=path,
        ok=ok,
        n_rows=n_rows,
        n_cols=n_cols,
        missing_expected_columns=missing_expected,
        missing_pct_by_column=missing_pct,
    )


def is_atp_tour_year_file(filename: str) -> bool:
    """Distingue les fichiers annuels ATP Tour principaux (ex: 2023.csv) des
    fichiers challenger/quali/wta/ongoing, pour ne les soumettre qu'eux au
    contrôle strict des colonnes de stats de match."""
    stem = filename.replace(".csv", "")
    return stem.isdigit()


# ---------------------------------------------------------------------------
# Rapports
# ---------------------------------------------------------------------------

def write_integrity_report(results: list[FileCheckResult]) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / "fetch_integrity_report.md"

    lines = [
        "# Rapport d'intégrité — fetch_data.py",
        f"\nGénéré le {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC\n",
        "| Fichier | Statut | Lignes | Colonnes | Colonnes attendues manquantes |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r.filename):
        statut = "OK" if r.ok else f"❌ {r.error or 'colonnes manquantes'}"
        missing_cols = ", ".join(sorted(r.missing_expected_columns)) if r.missing_expected_columns else "—"
        lines.append(f"| {r.filename} | {statut} | {r.n_rows} | {r.n_cols} | {missing_cols} |")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n→ Rapport d'intégrité écrit : {report_path}")
    return report_path


def write_missing_values_report(results: list[FileCheckResult]) -> Path:
    """Documente le % de valeurs manquantes par colonne, par fichier annuel
    (cf. Étape 1 du README : 'Documenter le pourcentage de valeurs manquantes
    par colonne, par année')."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / "missing_values_report.csv"

    year_results = [r for r in results if is_atp_tour_year_file(r.filename) and r.n_rows > 0]
    all_columns: set[str] = set()
    for r in year_results:
        all_columns |= set(r.missing_pct_by_column.keys())
    all_columns = sorted(all_columns)

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", *all_columns])
        for r in sorted(year_results, key=lambda r: r.filename):
            year = r.filename.replace(".csv", "")
            row = [year] + [r.missing_pct_by_column.get(col, "") for col in all_columns]
            writer.writerow(row)

    print(f"→ Rapport de valeurs manquantes écrit : {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Télécharge les données ATP Tour depuis TennisMyLife.")
    parser.add_argument(
        "--years", type=int, nargs="+", default=DEFAULT_YEARS,
        help="Années à télécharger (défaut: 2010-2026).",
    )
    parser.add_argument(
        "--include-challenger", action="store_true",
        help="Télécharge aussi les CSV ATP Challenger Tour.",
    )
    parser.add_argument(
        "--include-quali", action="store_true",
        help="Télécharge aussi les CSV ATP Tour Qualifying.",
    )
    parser.add_argument(
        "--include-consolidated", action="store_true",
        help="Télécharge aussi ATP_Database.csv (schéma différent, 12 colonnes "
             "consolidées seulement — exclu par défaut, voir commentaire dans build_target_urls).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("ÉTAPE 1 — Collecte des données ATP (TennisMyLife)")

    # On interroge l'API pour information/traçabilité, mais on télécharge
    # via des URLs construites explicitement pour garder le contrôle exact
    # de ce qui est récupéré 
    fetch_file_list()

    targets = build_target_urls(
        years=args.years,
        include_challenger=args.include_challenger,
        include_quali=args.include_quali,
    )
    if args.include_consolidated:
        targets["ATP_Database.csv"] = f"{BASE_DATA_URL}/ATP_Database.csv"

    print(f"\n→ Téléchargement de {len(targets)} fichiers vers {RAW_DIR}/")
    log = download_files(targets)

    n_ok = sum(1 for _, ok, _ in log if ok)
    n_fail = len(log) - n_ok
    print(f"\n→ Téléchargements terminés : {n_ok} réussis, {n_fail} échoués.")

    print("\n→ Vérification d'intégrité des fichiers...")
    results = [check_file(RAW_DIR / name) for name in targets]
    for r in results:
        if r.ok:
            print(f"  ✓ {r.filename}: {r.n_rows} lignes, {r.n_cols} colonnes — OK")
        else:
            reason = r.error or f"colonnes manquantes: {sorted(r.missing_expected_columns)}"
            print(f"  ✗ {r.filename}: {reason}")

    write_integrity_report(results)
    write_missing_values_report(results)

    print("\n" + "=" * 70)
    if n_fail == 0 and all(r.ok for r in results if is_atp_tour_year_file(r.filename)):
        print("Étape 1 terminée avec succès.")
        return 0
    else:
        print("⚠ Étape 1 terminée avec des avertissements — voir le rapport d'intégrité.")
        return 1


if __name__ == "__main__":
    sys.exit(main())