#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 2.

BUG CORRIGÉ (important) : le split A/B de la v1 faisait `df.iloc[:mid]` /
`df.iloc[mid:]` en supposant que le CSV chargé était déjà trié par ordre
chronologique. Or le script 8 (v1) sauvegardait son fichier trié par
`edge_abs` décroissant -> le split A/B ne correspondait probablement PAS à
deux périodes temporelles indépendantes, ce qui invalidait le test lui-même.
Ici, on exige une colonne `tourney_date` et on trie explicitement dessus
avant tout split.

CHANGEMENT MÉTHODOLOGIQUE : on ne teste plus le même pourcentage (top 30%)
sur A et sur B en le "redécouvrant" à chaque fois. On sélectionne le
meilleur pourcentage / seuil UNIQUEMENT sur A (période de sélection), on le
gèle, et on l'applique tel quel sur B (holdout). C'est le seul chiffre qui
doit être interprété comme une estimation de performance out-of-sample.

Ajout : intervalle de confiance bootstrap sur le ROI, en plus de la
p-value de permutation déjà présente en v1.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "evaluate model"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# qu'il contienne au minimum : tourney_date, confidence, PSW, bet_won.
RESULTS_FILE = PROJECT_ROOT / "outputs" /"test roi"/ "sliding_confidence_novig_flat_full.csv"

PERCENTAGES_GRID = (10, 20, 30, 40, 50, 70, 100)
MIN_BETS_FOR_SELECTION = 60


def roi_on_subset(df):
    if len(df) == 0:
        return np.nan, 0
    n = len(df)
    profit = (df.loc[df["bet_won"] == 1, "PSW"].sum() - n) / n
    return profit * 100, n


def bootstrap_ci(df, n_boot=3000, seed=0, ci=(2.5, 97.5)):
    if len(df) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(df)
    rois = np.empty(n_boot)
    for i in range(n_boot):
        sample = df.iloc[rng.integers(0, n, size=n)]
        roi, _ = roi_on_subset(sample)
        rois[i] = roi
    lo, hi = np.nanpercentile(rois, ci)
    return np.nanmean(rois), lo, hi


def permutation_pvalue(df, n_permutations=10000, seed=0):
    if len(df) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    odds = df["PSW"].values
    outcomes = df["bet_won"].values
    k = len(df)
    actual_roi, _ = roi_on_subset(df)
    random_rois = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(outcomes)
        profit = (np.sum(odds[shuffled == 1]) - k) / k
        random_rois[i] = profit * 100
    return np.sum(random_rois >= actual_roi) / n_permutations


def roi_by_confidence_decile(df):
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    n = len(df)
    rows = []
    for d in range(10):
        lo, hi = int(n * d / 10), int(n * (d + 1) / 10)
        subset = df.iloc[lo:hi]
        roi, count = roi_on_subset(subset)
        avg_conf = subset["confidence"].mean() if count else np.nan
        rows.append({"decile": f"{d*10}-{(d+1)*10}%", "n_bets": count, "avg_confidence": avg_conf, "roi_pct": roi})
    return pd.DataFrame(rows)


def select_best_percentage(df_a, sort_col="confidence", percentages=PERCENTAGES_GRID, min_bets=MIN_BETS_FOR_SELECTION):
    """Choisit le % qui maximise la borne basse bootstrap, sur A UNIQUEMENT."""
    df_a_sorted = df_a.sort_values(sort_col, ascending=False).reset_index(drop=True)
    n = len(df_a_sorted)
    best = None
    for pct in percentages:
        k = int(n * pct / 100)
        if k < min_bets:
            continue
        subset = df_a_sorted.head(k)
        mean_roi, lo, hi = bootstrap_ci(subset)
        cand = {"percentage": pct, "n_bets": k, "roi_ci_low": lo, "roi_ci_high": hi, "roi_mean_boot": mean_roi}
        if best is None or cand["roi_ci_low"] > best["roi_ci_low"]:
            best = cand
    return best


def calibration_table(df, n_bins=10):
    df = df.copy()
    df["implied_prob_chosen"] = 1.0 / df["PSW"]
    df["model_prob_est"] = df["confidence"] * df["implied_prob_chosen"]
    df["model_prob_est"] = df["model_prob_est"].clip(0, 1)
    df["bin"] = pd.qcut(df["model_prob_est"], q=n_bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(
        n=("bet_won", "size"),
        predicted_prob=("model_prob_est", "mean"),
        observed_freq=("bet_won", "mean"),
    ).reset_index()


def plot_calibration(table, path):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Calibration parfaite")
    plt.scatter(table["predicted_prob"], table["observed_freq"], s=table["n"] / table["n"].max() * 300)
    plt.xlabel("Probabilité prédite (implicite)")
    plt.ylabel("Fréquence observée")
    plt.title("Calibration du modèle sur le test set")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.savefig(path)
    plt.close()


def main():
    if not RESULTS_FILE.exists():
        print(f"✗ {RESULTS_FILE} introuvable — lance d'abord le script 8 (v2).")
        return

    df = pd.read_csv(RESULTS_FILE)
    if "tourney_date" not in df.columns:
        print("✗ Colonne 'tourney_date' absente : le split A/B chronologique n'est pas fiable sans elle.")
        print("  -> régénère ce fichier avec le script 8 v2, qui la propage désormais.")
        return

    df["tourney_date"] = df["tourney_date"].astype(str)
    df = df.sort_values("tourney_date", kind="mergesort").reset_index(drop=True)  # tri EXPLICITE, plus de dépendance à l'ordre du fichier
    print(f"→ {len(df)} paris chargés depuis {RESULTS_FILE.name}, triés par tourney_date.\n")

    # --- FILTRAGE DES COTES MANQUANTES/INVALIDES ------------------------
    # Le fichier "_full" produit par le script 8 contient TOUS les matchs du
    # test, y compris ceux sans cote Pinnacle exploitable (PSW/PSL absents,
    # remplis à 0 par fillna). Ces lignes ont une confidence forcée à 0 par
    # la division sécurisée (0/0 -> 0), PAS une vraie confiance faible.
    # Sans ce filtre, elles se trient en fin de classement et contaminent
    # tout seuil "top X%" qui dépasse ~60% -> à exclure systématiquement
    # avant toute sélection ou tout calcul de ROI/décile.
    n_before = len(df)
    df = df[(df["PSW"] > 0) & (df["confidence"] > 0)].reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"⚠️ {n_removed} paris exclus (cote manquante ou confidence=0 artificielle) "
              f"sur {n_before} chargés -> {len(df)} paris valides conservés.\n")


    print("=== ROI par décile de confiance (non cumulatif, référence descriptive) ===")
    decile_table = roi_by_confidence_decile(df)
    print(decile_table.to_string(index=False))
    decile_table.to_csv(OUTPUTS_DIR / "roi_by_decile.csv", index=False)

    # --- Split chronologique A (sélection) / B (holdout gelé) ---------
    mid = len(df) // 2
    df_a = df.iloc[:mid].reset_index(drop=True)
    df_b = df.iloc[mid:].reset_index(drop=True)
    print(f"\n=== Split chronologique : A = {len(df_a)} paris (sélection), B = {len(df_b)} paris (holdout) ===")

    best = select_best_percentage(df_a)
    if best is None:
        print("✗ Aucun seuil n'atteint le minimum de paris sur A.")
        return
    print(f"Meilleur seuil choisi SUR A UNIQUEMENT : top {best['percentage']}% "
          f"(n={best['n_bets']}, IC bootstrap sur A = [{best['roi_ci_low']:.2f}% ; {best['roi_ci_high']:.2f}%])")

    df_b_sorted = df_b.sort_values("confidence", ascending=False).reset_index(drop=True)
    k_b = int(len(df_b_sorted) * best["percentage"] / 100)
    holdout_bets = df_b_sorted.head(k_b)
    holdout_roi, n_holdout = roi_on_subset(holdout_bets)
    boot_mean, boot_lo, boot_hi = bootstrap_ci(holdout_bets)
    p_value = permutation_pvalue(holdout_bets)

    print(f"\n=== Application GELÉE du seuil top {best['percentage']}% sur B (jamais consulté pendant la sélection) ===")
    print(f"ROI observé sur B : {holdout_roi:.2f}% (n={n_holdout})")
    print(f"IC bootstrap 95% : [{boot_lo:.2f}% ; {boot_hi:.2f}%]")
    print(f"P-value (permutation) : {p_value:.4f}")
    if p_value < 0.05 and boot_lo > 0:
        print("🟢 Edge potentiellement réel sur cette période holdout.")
    else:
        print("🔴 Pas de preuve suffisante d'un edge réel (voir p-value / IC).")

    print("\n=== Calibration du modèle (sur tout l'échantillon, à titre descriptif) ===")
    calib = calibration_table(df)
    print(calib.to_string(index=False))
    plot_calibration(calib, OUTPUTS_DIR / "calibration_plot.png")
    print(f"📊 Graphique de calibration sauvegardé dans {OUTPUTS_DIR / 'calibration_plot.png'}")

    with open(OUTPUTS_DIR / "validation_summary.txt", "w") as f:
        f.write("SPLIT A (sélection) / B (holdout gelé), chronologique, trié explicitement sur tourney_date\n")
        f.write(f"Seuil retenu sur A : top {best['percentage']}% (n={best['n_bets']})\n")
        f.write(f"ROI sur B (holdout, jamais vu pendant la sélection) : {holdout_roi:.2f}% (n={n_holdout})\n")
        f.write(f"IC bootstrap 95% sur B : [{boot_lo:.2f}% ; {boot_hi:.2f}%]\n")
        f.write(f"P-value permutation sur B : {p_value:.4f}\n\n")
        f.write("ROI PAR DECILE (descriptif, tout l'échantillon)\n")
        f.write(decile_table.to_string(index=False))

    print(f"\n✅ Résumé écrit dans {OUTPUTS_DIR / 'validation_summary.txt'}")


if __name__ == "__main__":
    main()