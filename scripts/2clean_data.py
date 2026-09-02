"""
clean_data.py
=============
Étape 2 du projet : concaténation et préparation des données.

Ce script :
1. Charge tous les CSV annuels ATP Tour présents dans data/raw/ (fichiers
   dont le nom est une année + ongoing_tourneys.csv
2. Concatène le tout en une table unique.
3. Convertit `tourney_date` en date exploitable et trie strictement par
   (tourney_date, match_num) — condition nécessaire à tout calcul
   chronologique ultérieur (Elo, moyennes glissantes, H2H).
4. Normalise les IDs joueurs (`winner_id`, `loser_id`) en chaînes de
   caractères propres (évite les soucis de type mixte int/str/NaN).
6. Ajoute une colonne `surface_norm` normalisée (valeurs harmonisées,
   casse uniforme) sans supprimer aucune ligne.
7. Écrit le résultat dans data/processed/atp_matches_all.csv et un rapport
   de synthèse dans outputs/.

Usage:
    python scripts/clean_data.py
    python scripts/clean_data.py --include-ongoing=false
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT/ "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT  / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT  / "outputs" / "cleaning"

OUTPUT_FILE = PROCESSED_DIR / "atp_matches_all.csv"

# Colonnes que l'on force à exister en sortie (valeur vide si absente en entrée),
# pour garantir un schéma stable même si une année a des colonnes en plus/moins.
CANONICAL_COLUMNS_ORDER = [
    "tourney_id", "tourney_name", "surface", "surface_norm", "draw_size",
    "tourney_level", "tourney_date", "match_num", "indoor",
    "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand",
    "winner_ht", "winner_ioc", "winner_age", "winner_rank", "winner_rank_points",
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand",
    "loser_ht", "loser_ioc", "loser_age", "loser_rank", "loser_rank_points",
    "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
    "source_file",
]

# Normalisation des libellés de surface : on harmonise la casse et les
# variantes connues, sans jamais supprimer une ligne dont la surface est
# absente ou inconnue (on la marque juste "Unknown").
SURFACE_NORMALIZATION = {
    "hard": "Hard",
    "clay": "Clay",
    "grass": "Grass",
    "carpet": "Carpet",
}


def is_year_file(path: Path) -> bool:
    return path.stem.isdigit()


def normalize_surface(raw_value: str) -> str:
    if not raw_value:
        return "Unknown"
    return SURFACE_NORMALIZATION.get(raw_value.strip().lower(), raw_value.strip() or "Unknown")


def parse_tourney_date(raw_value: str) -> str:
    """
    Convertit une date au format YYYYMMDD (format ATP standard) en YYYY-MM-DD.
    Retourne une chaîne vide si le parsing échoue (ligne conservée quand même,
    mais elle sera visible dans le rapport comme date invalide).
    """
    raw_value = (raw_value or "").strip()
    if len(raw_value) != 8 or not raw_value.isdigit():
        return ""
    try:
        dt = datetime.strptime(raw_value, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return ""


def safe_int_str(raw_value: str) -> str:
    """Normalise un ID/numéro potentiellement flottant ('104925.0') en
    chaîne entière propre ('104925'), sans planter sur des valeurs vides."""
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return ""
    try:
        return str(int(float(raw_value)))
    except ValueError:
        return raw_value  # on garde tel quel plutôt que de perdre l'info


def load_source_files(include_ongoing: bool) -> list[Path]:
    if not RAW_DIR.exists():
        return []
    files = [p for p in RAW_DIR.glob("*.csv") if is_year_file(p)]
    if include_ongoing:
        ongoing = RAW_DIR / "ongoing_tourneys.csv"
        if ongoing.exists():
            files.append(ongoing)
    return sorted(files)


def concatenate_and_clean(files: list[Path]) -> tuple[list[dict], Counter, list[str]]:
    """
    Charge et normalise toutes les lignes. Retourne :
    - la liste des lignes normalisées (dicts),
    - un compteur de lignes par fichier source (traçabilité),
    - une liste d'avertissements (ex: dates invalides) pour le rapport.
    """
    all_rows: list[dict] = []
    rows_per_file: Counter = Counter()
    warnings: list[str] = []

    for path in files:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            n_rows_this_file = 0
            n_bad_dates_this_file = 0
            for row in reader:
                normalized = {col: row.get(col, "") or "" for col in CANONICAL_COLUMNS_ORDER if col not in ("surface_norm", "source_file")}
                normalized["source_file"] = path.name

                # Surface : jamais de perte de ligne, juste normalisation d'affichage.
                normalized["surface_norm"] = normalize_surface(row.get("surface", ""))

                # Date : condition nécessaire au tri chronologique strict (principe méthodologique #1).
                parsed_date = parse_tourney_date(row.get("tourney_date", ""))
                if not parsed_date:
                    n_bad_dates_this_file += 1
                normalized["tourney_date"] = parsed_date or row.get("tourney_date", "")

                # IDs joueurs normalisés en chaînes propres pour fiabiliser les futures jointures H2H / Elo.
                normalized["winner_id"] = safe_int_str(row.get("winner_id", ""))
                normalized["loser_id"] = safe_int_str(row.get("loser_id", ""))
                normalized["match_num"] = safe_int_str(row.get("match_num", ""))

                all_rows.append(normalized)
                n_rows_this_file += 1

            rows_per_file[path.name] = n_rows_this_file
            if n_bad_dates_this_file:
                warnings.append(
                    f"{path.name}: {n_bad_dates_this_file} ligne(s) avec tourney_date invalide/non-standard "
                    f"(conservée(s) telle(s) quelle(s), à surveiller pour le tri chronologique)."
                )

    return all_rows, rows_per_file, warnings


def sort_chronologically(rows: list[dict]) -> list[dict]:
    """
    Tri strict par (tourney_date, match_num), condition impérative avant
    tout calcul d'Elo ou de feature glissante.
    Les dates invalides/vides sont placées en tête (traitées comme non
    ordonnables) pour qu'elles restent visibles plutôt que silencieusement
    mélangées au reste.
    """
    def sort_key(row: dict):
        date_str = row.get("tourney_date", "")
        match_num_str = row.get("match_num", "")
        date_key = date_str if len(date_str) == 10 else "0000-00-00"
        try:
            match_num_key = int(match_num_str)
        except ValueError:
            match_num_key = -1
        return (date_key, match_num_key)

    return sorted(rows, key=sort_key)


def write_output(rows: list[dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS_ORDER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_report(rows: list[dict], rows_per_file: Counter, warnings: list[str]) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / "clean_data_report.md"

    surface_counts = Counter(r["surface_norm"] for r in rows)
    date_range = ""
    valid_dates = sorted(r["tourney_date"] for r in rows if len(r.get("tourney_date", "")) == 10)
    if valid_dates:
        date_range = f"{valid_dates[0]} → {valid_dates[-1]}"

    lines = [
        "# Rapport de nettoyage — clean_data.py",
        f"\nGénéré le {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC\n",
        f"**Total matchs concaténés (toutes surfaces confondues) :** {len(rows)}",
        f"**Plage temporelle couverte :** {date_range or 'indéterminée'}\n",
        "## Lignes par fichier source",
        "| Fichier | Lignes |",
        "|---|---|",
    ]
    for name, n in sorted(rows_per_file.items()):
        lines.append(f"| {name} | {n} |")

    lines += [
        "\n## Répartition par surface (aucun match exclu)",
        "| Surface | Matchs |",
        "|---|---|",
    ]
    for surface, n in surface_counts.most_common():
        lines.append(f"| {surface} | {n} |")

    if warnings:
        lines += ["\n## Avertissements"]
        lines += [f"- {w}" for w in warnings]
    else:
        lines += ["\n## Avertissements", "- Aucun."]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ Rapport de nettoyage écrit : {report_path}")
    return report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatène et nettoie les CSV ATP Tour bruts.")
    parser.add_argument(
        "--include-ongoing", type=lambda s: s.lower() != "false", default=True,
        help="Inclure ongoing_tourneys.csv dans la concaténation (défaut: true).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("ÉTAPE 2 — Nettoyage et concaténation des données")

    files = load_source_files(include_ongoing=args.include_ongoing)
    if not files:
        print(f"✗ Aucun fichier annuel trouvé dans {RAW_DIR}/. "
              f"Lance d'abord scripts/fetch_data.py.")
        return 1

    print(f"→ {len(files)} fichier(s) source trouvé(s) :")
    for f in files:
        print(f"  - {f.name}")

    rows, rows_per_file, warnings = concatenate_and_clean(files)
    print(f"\n→ {len(rows)} lignes concaténées (tous matchs, toutes surfaces, aucun filtre).")

    rows = sort_chronologically(rows)
    print("→ Tri chronologique strict appliqué (tourney_date, match_num).")

    write_output(rows)
    print(f"→ Fichier écrit : {OUTPUT_FILE}")

    write_summary_report(rows, rows_per_file, warnings)

    if warnings:
        print(f"\n⚠ {len(warnings)} avertissement(s) — voir outputs/clean_data_report.md")

    print("Étape 2 terminée.")
    return 0


if __name__ == "__main__":
    sys.exit(main())