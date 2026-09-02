#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcul et ajout des features de Face-à-Face (Head-to-Head / H2H) sans Data Leakage.

Features générées :
  1. h2h_count : Nombre total de confrontations passées entre les deux joueurs.
  2. h2h_smoothed_diff : Avantage lissé du vainqueur sur le perdant (toutes surfaces).
  3. h2h_surface_smoothed_diff : Avantage lissé du vainqueur sur cette surface spécifique.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "confrontation"

INPUT_FILE = PROCESSED_DIR / "atp_matches_with_fatigue.csv"
OUTPUT_FILE = PROCESSED_DIR / "atp_matches_with_h2h.csv"
REPORT_FILE = OUTPUTS_DIR / "confrontation_report.md"


def main():
    print("1. Chargement du dataset...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"   --> {len(df)} matchs chargés depuis {INPUT_FILE.name}")

    # Index d'origine pour restaurer l'ordre exact à la fin
    df["orig_idx"] = np.arange(len(df))

    print("2. Tri chronologique rigoureux (anti-Data Leakage)...")
    # Conversion de la date en clé textuelle YYYYMMDD pour un tri fiable
    df["date_sort_key"] = df["tourney_date"].astype(str).str.replace("-", "")
    
    # Tri par date, puis par numéro de match
    df_sorted = df.sort_values(
        by=["date_sort_key", "match_num", "orig_idx"], ascending=True
    ).copy()

    print("3. Calcul des métriques H2H en boucle séquentielle...")

    # Structure de données pour stocker l'historique au fil de l'eau
    # h2h_store[(player_a, player_b)] = { ... }
    h2h_store = defaultdict(
        lambda: {
            "total": 0,
            "wins": defaultdict(int),
            "surface_total": defaultdict(int),
            "surface_wins": defaultdict(lambda: defaultdict(int)),
        }
    )

    h2h_counts = []
    h2h_diffs = []
    h2h_surface_diffs = []

    # Parcours séquentiel
    for row in df_sorted.itertuples():
        w_id = str(row.winner_id).strip()
        l_id = str(row.loser_id).strip()

        # Récupération de la surface
        surf = getattr(row, "surface_norm", None)
        if pd.isna(surf) or not surf:
            surf = getattr(row, "surface", "Unknown")
        surf = str(surf).strip()

        # Clé unique pour la paire de joueurs (indépendante de l'ordre)
        pair_key = tuple(sorted([w_id, l_id]))
        record = h2h_store[pair_key]

        # -------------------------------------------------------------
        # ÉTAPE A : LECTURE DE L'ÉTAT AVANT LE MATCH (ZERO DATA LEAKAGE)
        # -------------------------------------------------------------
        n_total = record["total"]
        w_wins = record["wins"][w_id]
        
        # Lissage bayésien global : (wins + 1) / (total + 2) - 0.5
        smoothed_diff = (w_wins + 1.0) / (n_total + 2.0) - 0.5

        # Lissage bayésien sur la surface spécifique
        n_surf_total = record["surface_total"][surf]
        w_surf_wins = record["surface_wins"][surf][w_id]
        surf_smoothed_diff = (
            (w_surf_wins + 1.0) / (n_surf_total + 2.0) - 0.5
        )

        h2h_counts.append(n_total)
        h2h_diffs.append(smoothed_diff)
        h2h_surface_diffs.append(surf_smoothed_diff)

        # -------------------------------------------------------------
        # ÉTAPE B : MISE À JOUR DE L'HISTORIQUE APRÈS LE MATCH
        # -------------------------------------------------------------
        record["total"] += 1
        record["wins"][w_id] += 1
        record["surface_total"][surf] += 1
        record["surface_wins"][surf][w_id] += 1

    # Affectation des nouvelles colonnes
    df_sorted["h2h_count"] = h2h_counts
    df_sorted["h2h_smoothed_diff"] = h2h_diffs
    df_sorted["h2h_surface_smoothed_diff"] = h2h_surface_diffs

    print("4. Restitution de l'ordre d'origine...")
    df_result = (
        df_sorted.sort_values(by="orig_idx")
        .drop(columns=["orig_idx", "date_sort_key"])
        .reset_index(drop=True)
    )

    print(f"5. Sauvegarde dans {OUTPUT_FILE}...")
    df_result.to_csv(OUTPUT_FILE, index=False)

    # ==========================================
    # ÉTAPE DE VÉRIFICATION & CRÉATION DU RAPPORT
    # ==========================================
    total_matches = len(df_result)
    matches_with_h2h = (df_result["h2h_count"] > 0).sum()
    pct_with_h2h = (matches_with_h2h / total_matches) * 100
    max_h2h = df_result["h2h_count"].max()
    matches_with_surface_h2h = (
        (df_result["h2h_count"] > 0)
        & (df_result["h2h_surface_smoothed_diff"] != 0.0)
    ).sum()

    lines = [
        "=" * 50,
        "🔎 RAPPORT DE GÉNÉRATION DES FEATURES H2H :",
        "=" * 50,
        f"• Nombre total de matchs traités : {total_matches}",
        f"• Matchs avec au moins 1 duel passé (h2h_count > 0) : {matches_with_h2h} ({pct_with_h2h:.2f}%)",
        f"• Matchs avec historique sur la même surface : {matches_with_surface_h2h} ({(matches_with_surface_h2h/total_matches)*100:.2f}%)",
        f"• Nombre maximum de duels enregistrés pour une paire : {max_h2h}",
        "",
        "Distribution des valeurs H2H Count :",
        f"{df_result['h2h_count'].value_counts().head(6).to_string()}",
        "",
        "SUCCÈS : Features confrontation créées sans Data Leakage !",
        "=" * 50,
    ]

    report_content = "\n".join(lines)
    print("\n" + report_content)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content + "\n")

    print(f"\n📄 Rapport sauvegardé dans : {REPORT_FILE}")


if __name__ == "__main__":
    main()