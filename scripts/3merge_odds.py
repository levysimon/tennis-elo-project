from __future__ import annotations
import glob
import re
import unicodedata
import warnings
from pathlib import Path
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_ODDS_DIR = PROJECT_ROOT / "data" / "betting"

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "merge matchs and odds"
INPUT_ATP_FILE = PROCESSED_DIR / "atp_matches_all.csv"
OUTPUT_MERGED_FILE = PROCESSED_DIR / "atp_matches_with_odds.csv"
OUTPUT_REPORT_FILE = OUTPUTS_DIR / "merge_report.md"


def clean_string(text: str) -> str:
    if pd.isna(text):
        return ""
    text = (
        unicodedata.normalize("NFD", str(text))
        .encode("ascii", "ignore")
        .decode("utf-8")
    )
    text = text.lower().replace("-", " ").replace(".", "")
    return re.sub(r"\s+", " ", text).strip()


def get_player_key_atp(name: str) -> str:
    cleaned = clean_string(name)
    parts = cleaned.split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    initial = parts[0][0]
    lastname = " ".join(parts[1:])
    return f"{initial} {lastname}"


def get_player_key_bet(name: str) -> str:
    cleaned = clean_string(name)
    parts = cleaned.split()
    if not parts:
        return ""
    last_part = parts[-1]
    if len(last_part) <= 3 and last_part.isalpha():
        initial = last_part[0]
        lastname = " ".join(parts[:-1])
    else:
        initial = parts[0][0]
        lastname = " ".join(parts[1:])
    return f"{initial} {lastname}"


def create_matchup_signature(p1_key: str, p2_key: str) -> str:
    sorted_players = sorted([p1_key, p2_key])
    return f"{sorted_players[0]}__vs__{sorted_players[1]}"


YEAR_MIN, YEAR_MAX = 1990, 2035


def extract_year(value) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (pd.Timestamp, datetime, date)):
        y = value.year
        return y if YEAR_MIN <= y <= YEAR_MAX else None

    if isinstance(value, (int, np.integer, float, np.floating)):
        v = int(value)
        s = str(v)
        if len(s) >= 6:
            y = int(s[:4])
            if YEAR_MIN <= y <= YEAR_MAX:
                return y
        try:
            d = date(1899, 12, 30) + timedelta(days=v)
            if YEAR_MIN <= d.year <= YEAR_MAX:
                return d.year
        except (OverflowError, ValueError):
            pass
        return None

    s = str(value).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None

    m = re.match(r"^(\d{4})", s)
    if m:
        y = int(m.group(1))
        if YEAR_MIN <= y <= YEAR_MAX:
            return y

    m = re.search(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$", s)
    if m:
        y = int(m.group(3))
        if y < 100:
            y += 2000 if y < 70 else 1900
        if YEAR_MIN <= y <= YEAR_MAX:
            return y

    return None


def load_all_odds_files(odds_dir: Path) -> pd.DataFrame:
    files = list(odds_dir.glob("*.xlsx")) + list(odds_dir.glob("*.xls"))
    dfs = []
    print(f"   --> {len(files)} fichiers d'odds trouvés dans {odds_dir}")
    for f in files:
        if f.suffix.lower() == ".xls":
            df = pd.read_excel(f, engine="xlrd")
        else:
            df = pd.read_excel(f, engine="openpyxl")
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"Aucun fichier trouvé dans {odds_dir}")
    return pd.concat(dfs, ignore_index=True)


def main():
    print("1. Chargement du dataset ATP...")
    df_atp = pd.read_csv(INPUT_ATP_FILE, low_memory=False)
    print(f"   --> {len(df_atp)} matchs ATP chargés.")

    print("2. Chargement des fichiers de paris...")
    df_odds = load_all_odds_files(RAW_ODDS_DIR)
    print(f"   --> {len(df_odds)} lignes de paris chargées.")

    print("3. Extraction des années (sans pd.to_datetime, pure Python)...")
    df_atp["year"] = df_atp["tourney_date"].apply(extract_year)
    df_odds["year"] = df_odds["Date"].apply(extract_year)

    n_year_missing_atp = df_atp["year"].isna().sum()
    n_year_missing_odds = df_odds["year"].isna().sum()
    if n_year_missing_atp:
        print(f"   ⚠️ {n_year_missing_atp} dates ATP non reconnues (year manquante) — vérifier leur format brut.")
    if n_year_missing_odds:
        print(f"   ⚠️ {n_year_missing_odds} dates de paris non reconnues (year manquante) — vérifier leur format brut.")

    df_atp["w_key"] = df_atp["winner_name"].apply(get_player_key_atp)
    df_atp["l_key"] = df_atp["loser_name"].apply(get_player_key_atp)
    df_atp["matchup_id"] = df_atp.apply(
        lambda r: create_matchup_signature(r["w_key"], r["l_key"]), axis=1
    )

    df_odds["w_key"] = df_odds["Winner"].apply(get_player_key_bet)
    df_odds["l_key"] = df_odds["Loser"].apply(get_player_key_bet)
    df_odds["matchup_id"] = df_odds.apply(
        lambda r: create_matchup_signature(r["w_key"], r["l_key"]), axis=1
    )

    odds_cols = [
        c
        for c in [
            "B365W", "B365L",
            "PSW", "PSL",
            "EXW", "EXL",
            "MaxW", "MaxL",
            "AvgW", "AvgL",
        ]
        if c in df_odds.columns
    ]

    print("4. Dédoublonnage et Fusion...")
    df_odds_clean = df_odds.drop_duplicates(
        subset=["year", "matchup_id"], keep="first"
    )

    merged = pd.merge(
        df_atp,
        df_odds_clean[["year", "matchup_id"] + odds_cols],
        on=["year", "matchup_id"],
        how="left",
    )

    merged.drop(
        columns=["w_key", "l_key", "matchup_id", "year"],
        inplace=True,
        errors="ignore",
    )

    print(f"5. Sauvegarde dans {OUTPUT_MERGED_FILE}...")
    merged.to_csv(OUTPUT_MERGED_FILE, index=False)

    # ==========================================
    # ÉTAPE DE VÉRIFICATION & CRÉATION DU RAPPORT
    # ==========================================
    total_matches = len(merged)
    odds_found_count = merged["B365W"].notna().sum() if "B365W" in merged.columns else 0
    match_rate = (odds_found_count / total_matches) * 100 if total_matches > 0 else 0

    lines = [
        "=" * 50,
        "🔎 RAPPORT DE VÉRIFICATION DU MERGE :",
        "=" * 50,
        f"• Nombre total de matchs ATP : {total_matches}",
        f"• Nombre de matchs avec cotes retrouvées (B365W) : {odds_found_count}",
        f"• Taux de correspondance (Match Rate) : {match_rate:.2f}%",
    ]

    if odds_found_count == 0:
        lines.append("\n❌ ERREUR : Aucune cote n'a été liée aux matchs !")
        lines.append("Vérifiez le chemin du dossier `RAW_ODDS_DIR` ou les noms des colonnes de vos fichiers Excel.")
    elif match_rate < 50:
        lines.append("\n⚠️ AVERTISSEMENT : Le taux de correspondance est anormalement bas (< 50%).")
    else:
        lines.append("\n✅ SUCCÈS : Le merge a fonctionné avec succès !")

    lines.append("=" * 50)

    report_content = "\n".join(lines)

    # Affichage dans la console
    print("\n" + report_content)

    # Sauvegarde du rapport texte dans le dossier outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content + "\n")

    print(f"\n📄 Rapport de merge sauvegardé dans : {OUTPUT_REPORT_FILE}")


if __name__ == "__main__":
    main()