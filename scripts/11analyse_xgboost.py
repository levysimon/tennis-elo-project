#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 2.

CHANGEMENT PRINCIPAL : la v1 comparait modèle complet vs baseline Elo sur
UN SEUL split (train <= 2023, test >= 2024). Un seul split donne un seul
chiffre, sans mesure d'incertitude ni de stabilité dans le temps. Ici on
répète la comparaison sur plusieurs origines de test glissantes (walk-
forward multi-origines) et on rapporte :
  - la moyenne et l'écart-type de l'écart d'accuracy / log-loss entre le
    modèle complet et la baseline Elo, à travers les périodes,
  - un intervalle de confiance bootstrap sur cet écart moyen.
Cela permet de répondre à la question "le gain du modèle complet sur Elo
est-il stable dans le temps, ou est-ce un artefact d'un split particulier ?"

DATA_START est remonté à 2002 (si votre dataset couvre bien cette période)
pour donner davantage de matchs d'entraînement à chaque origine, sans rien
changer à la logique de séparation temporelle stricte (toujours entraîné
uniquement sur le passé de chaque origine de test).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "atp_modeling_dataset.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "analyse xgboost"

DATA_START = "2000-01-01"
# Origines de test successives : chaque origine définit une coupure
# train(< cutoff) / test([cutoff, cutoff + TEST_WINDOW_DAYS jours équivalent
# en nombre de lignes approximatif, ici on utilise directement des dates)).
TEST_ORIGINS = ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]
TEST_WINDOW_END = "2026-12-31"  # chaque test va de son origine jusqu'à la fin des données

ODDS_LEAKAGE_PREFIXES = (
    'player1_b365', 'player1_ps', 'player1_ex', 'player1_max', 'player1_avg',
    'player2_b365', 'player2_ps', 'player2_ex', 'player2_max', 'player2_avg',
)
ODDS_LEAKAGE_EXACT = ('player1_elo_win_prob',)

DROP_COLS = [
    'target_player1_won', 'tourney_date', 'tourney_id', 'match_num',
    'dataset_split', 'minutes', 'l_1stIn', 'w_1stIn', 'l_svpt', 'w_svpt'
]


def get_feature_cols(df):
    candidate_cols = [
        c for c in df.columns
        if c not in DROP_COLS
        and c not in ODDS_LEAKAGE_EXACT
        and not c.startswith("player1_name")
        and not c.startswith("player2_name")
        and not c.startswith(ODDS_LEAKAGE_PREFIXES)
    ]
    return df[candidate_cols].select_dtypes(include=[np.number, bool]).columns.tolist()


def train_and_eval_one_origin(df, cutoff, feature_cols):
    train = df[df["tourney_date"] < cutoff]
    test = df[(df["tourney_date"] >= cutoff) & (df["tourney_date"] <= TEST_WINDOW_END)]
    if len(train) < 500 or len(test) < 100:
        return None

    dtrain = xgb.DMatrix(train[feature_cols].fillna(0), label=train["target_player1_won"])
    dtest = xgb.DMatrix(test[feature_cols].fillna(0), label=test["target_player1_won"])
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "max_depth": 5, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
    }
    model = xgb.train(params, dtrain, num_boost_round=200)
    proba_full = model.predict(dtest)
    y_test = test["target_player1_won"].values

    full_metrics = {
        "accuracy": accuracy_score(y_test, proba_full > 0.5),
        "log_loss": log_loss(y_test, proba_full),
        "brier": brier_score_loss(y_test, proba_full),
    }

    elo_metrics = None
    needed = ["player1_elo_pre", "player2_elo_pre", "player1_surface_elo_pre", "player2_surface_elo_pre"]
    if all(c in df.columns for c in needed):
        train_elo_diff = (train["player1_elo_pre"] - train["player2_elo_pre"]).fillna(0)
        train_surf_diff = (train["player1_surface_elo_pre"] - train["player2_surface_elo_pre"]).fillna(0)
        test_elo_diff = (test["player1_elo_pre"] - test["player2_elo_pre"]).fillna(0)
        test_surf_diff = (test["player1_surface_elo_pre"] - test["player2_surface_elo_pre"]).fillna(0)

        X_train = np.column_stack([train_elo_diff, train_surf_diff])
        X_test = np.column_stack([test_elo_diff, test_surf_diff])
        clf = LogisticRegression()
        clf.fit(X_train, train["target_player1_won"])
        proba_elo = clf.predict_proba(X_test)[:, 1]
        elo_metrics = {
            "accuracy": accuracy_score(y_test, proba_elo > 0.5),
            "log_loss": log_loss(y_test, proba_elo),
            "brier": brier_score_loss(y_test, proba_elo),
        }

    return {
        "cutoff": cutoff, "n_train": len(train), "n_test": len(test),
        "full": full_metrics, "elo": elo_metrics,
    }


def bootstrap_ci_on_diffs(diffs, n_boot=5000, seed=0):
    """IC bootstrap sur la moyenne d'une série de différences (une par origine)."""
    diffs = np.array([d for d in diffs if not np.isnan(d)])
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = diffs[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return diffs.mean(), lo, hi


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Lecture du CSV...")
    df = pd.read_csv(DATASET_FILE, low_memory=False)
    print(f"   {df.shape[0]} lignes, {df.shape[1]} colonnes")

    print("2. Tri par date (comparaison texte, format ISO)...")
    df["tourney_date"] = df["tourney_date"].astype(str)
    df = df[df["tourney_date"] >= DATA_START].copy()
    df = df.sort_values("tourney_date", kind="mergesort").reset_index(drop=True)
    print(f"   {len(df)} lignes utilisées à partir de {DATA_START}")

    feature_cols = get_feature_cols(df)
    print(f"   {len(feature_cols)} features candidates (cotes/marché exclues)")

    print("\n3. Backtest à origines multiples (walk-forward)...")
    per_origin_results = []
    for cutoff in TEST_ORIGINS:
        print(f"   → origine {cutoff}...")
        res = train_and_eval_one_origin(df, cutoff, feature_cols)
        if res is None:
            print(f"     ⚠️ Échantillon insuffisant pour cette origine, ignorée.")
            continue
        per_origin_results.append(res)
        acc_full = res["full"]["accuracy"]
        acc_elo = res["elo"]["accuracy"] if res["elo"] else float("nan")
        print(f"     n_train={res['n_train']}, n_test={res['n_test']}, "
              f"accuracy modèle={acc_full:.4f}, accuracy Elo={acc_elo:.4f}, "
              f"gain={acc_full - acc_elo:+.4f}")

    if not per_origin_results:
        print("✗ Aucune origine n'a produit de résultat exploitable.")
        return

    print("\n=== Stabilité du gain (modèle complet - Elo) à travers les origines ===")
    acc_diffs = [r["full"]["accuracy"] - r["elo"]["accuracy"] for r in per_origin_results if r["elo"]]
    logloss_diffs = [r["elo"]["log_loss"] - r["full"]["log_loss"] for r in per_origin_results if r["elo"]]  # positif = modèle meilleur

    acc_mean, acc_lo, acc_hi = bootstrap_ci_on_diffs(acc_diffs)
    ll_mean, ll_lo, ll_hi = bootstrap_ci_on_diffs(logloss_diffs)

    print(f"Gain d'accuracy moyen (modèle - Elo) : {acc_mean:+.4f}  |  IC bootstrap 95% : [{acc_lo:+.4f} ; {acc_hi:+.4f}]")
    print(f"Gain de log-loss moyen (Elo - modèle, positif = mieux) : {ll_mean:+.4f}  |  IC bootstrap 95% : [{ll_lo:+.4f} ; {ll_hi:+.4f}]")
    if acc_lo > 0:
        print("🟢 Le gain d'accuracy du modèle complet sur Elo semble robuste à travers les périodes testées.")
    else:
        print("🟡 L'IC du gain d'accuracy inclut 0 : le gain n'est pas garanti stable sur toutes les périodes — "
              "regarder le détail par origine ci-dessus avant de conclure.")

    # Importance des features sur le dernier modèle entraîné (origine la plus récente)
    last_train = df[df["tourney_date"] < TEST_ORIGINS[-1]]
    last_test = df[(df["tourney_date"] >= TEST_ORIGINS[-1]) & (df["tourney_date"] <= TEST_WINDOW_END)]
    if len(last_train) > 500 and len(last_test) > 100:
        dtrain = xgb.DMatrix(last_train[feature_cols].fillna(0), label=last_train["target_player1_won"])
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 5,
                  "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}
        model = xgb.train(params, dtrain, num_boost_round=200)
        gain = model.get_score(importance_type="gain")
        imp_table = pd.Series(gain).sort_values(ascending=False).head(50).rename("gain").to_frame()
        imp_table.to_csv(OUTPUTS_DIR / "feature_importance_last_origin.csv")
        print(f"\n📄 Importance des features (dernière origine) sauvegardée dans "
              f"{OUTPUTS_DIR / 'feature_importance_last_origin.csv'}")

    # Sauvegarde du résumé par origine + stabilité
    summary_rows = []
    for r in per_origin_results:
        row = {"cutoff": r["cutoff"], "n_train": r["n_train"], "n_test": r["n_test"],
               "accuracy_full": r["full"]["accuracy"], "log_loss_full": r["full"]["log_loss"]}
        if r["elo"]:
            row.update({"accuracy_elo": r["elo"]["accuracy"], "log_loss_elo": r["elo"]["log_loss"]})
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUTS_DIR / "walkforward_summary.csv", index=False)

    with open(OUTPUTS_DIR / "walkforward_summary.txt", "w") as f:
        f.write(summary_df.to_string(index=False))
        f.write(f"\n\nGain accuracy moyen : {acc_mean:+.4f}, IC95%: [{acc_lo:+.4f}; {acc_hi:+.4f}]\n")
        f.write(f"Gain log-loss moyen : {ll_mean:+.4f}, IC95%: [{ll_lo:+.4f}; {ll_hi:+.4f}]\n")

    print(f"\n✅ Résumé écrit dans {OUTPUTS_DIR / 'walkforward_summary.txt'} "
          f"et {OUTPUTS_DIR / 'walkforward_summary.csv'}")


if __name__ == "__main__":
    main()