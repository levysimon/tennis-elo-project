#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 3.

CHANGEMENTS PAR RAPPORT A LA V2 :
1. Filtre d'exclusion des colonnes de cotes : passage d'une exclusion par
   PRÉFIXE (startswith) à une exclusion par NOM EXACT, identique au correctif
   déjà appliqué à modeling.py. L'ancien mécanisme fonctionnait avec le
   dataset actuel (vérifié : aucune des 54 colonnes engineered streak/
   momentum/quality_ema/delta_form/ema_long/r10_* n'était perdue), mais
   restait fragile : une future feature nommée 'player1_avg_XXX' ou
   'player1_max_XXX' aurait été silencieusement exclue sans avertissement.
2. SUPPRESSION de `.fillna(0)` sur les features avant construction des
   DMatrix. Plusieurs des nouvelles features (momentum_*, delta_form_*,
   r10_avg_opp_elo_beaten, quality_ema) ont 0 comme valeur RÉELLE et
   significative (ex: momentum nul = pas de tendance). Remplacer les NaN
   (= historique insuffisant, cf. build_features.py) par 0 rendait "pas de
   donnée" et "valeur réellement nulle" indiscernables pour le modèle,
   diluant le signal de ces features. XGBoost gère nativement les valeurs
   manquantes (apprentissage d'une direction de split par défaut) : on lui
   laisse les NaN tels quels.
3. Le tableau d'importance des features sauvegarde désormais la table
   COMPLÈTE (plus de head(50)), et une liste séparée des features JAMAIS
   utilisées dans un split (gain nul, absentes de model.get_score()) - les
   deux causes de "feature absente du tableau" étaient confondues avant.
5. Ajout d'une 3e baseline : le CLASSEMENT ATP BRUT (player{1,2}_rank et
   player{1,2}_rank_points), en plus de la baseline Elo — conformément au
   principe #5 du README ("Le Elo seul, ou même le classement ATP brut,
   doit servir de référence"). Construite en miroir de la baseline Elo :
   régression logistique sur 2 features (écart de rang, écart de points
   ATP), même traitement des NaN (joueurs non classés -> rang très bas
   fictif, points = 0, cf. RANK_UNRANKED_FALLBACK ci-dessous).
   NB : ce dataset est déjà exclusivement composé de matchs ATP Tour (cf.
   README section 2 — pas de Challenger/ITF mélangé), donc aucun filtrage
   supplémentaire n'est nécessaire pour isoler "l'ATP" : c'est déjà le cas
   sur l'ensemble du fichier.
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
TEST_ORIGINS = ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]
TEST_WINDOW_END = "2026-12-31"

# Valeur fictive utilisée pour un joueur NON CLASSÉ (wildcard, retour de
# blessure, etc.) : un rang très mauvais plutôt qu'un NaN, pour ne pas
# fausser l'écart de rang. 2000 est bien au-delà de tout rang ATP réaliste
# rencontré dans ce dataset (le rang le plus faible représenté reste très
# inférieur à ce seuil), donc un joueur non classé est toujours traité
# comme "beaucoup plus faible" que n'importe quel adversaire classé.
RANK_UNRANKED_FALLBACK = 2000.0

# --- Colonnes de cotes de bookmaker : EXCLUSION PAR NOM EXACT (cf. modeling.py) ---
BOOKMAKER_SUFFIXES = ('b365', 'ps', 'ex', 'max', 'avg')
ODDS_LEAKAGE_EXACT_COLS = {
    f"player{n}_{suffix}" for n in (1, 2) for suffix in BOOKMAKER_SUFFIXES
}
OTHER_LEAKAGE_EXACT_COLS = {'player1_elo_win_prob'}
ALL_EXACT_EXCLUDED_COLS = ODDS_LEAKAGE_EXACT_COLS | OTHER_LEAKAGE_EXACT_COLS

ENGINEERED_FEATURE_KEYWORDS = (
    "streak", "quality_ema", "momentum_", "delta_form_", "ema_long_",
    "r10_avg_opp_elo_beaten", "r10_upset_win_pct", "r10_bad_loss_pct",
)

DROP_COLS = [
    'target_player1_won', 'tourney_date', 'tourney_id', 'match_num',
    'dataset_split', 'minutes', 'l_1stIn', 'w_1stIn', 'l_svpt', 'w_svpt'
]


def get_feature_cols(df, verbose=False):
    candidate_cols = [
        c for c in df.columns
        if c not in DROP_COLS
        and c not in ALL_EXACT_EXCLUDED_COLS
        and c not in ('player1_name', 'player2_name')
    ]
    feature_cols = df[candidate_cols].select_dtypes(include=[np.number, bool]).columns.tolist()

    if verbose:
        engineered_in_dataset = [c for c in df.columns if any(k in c for k in ENGINEERED_FEATURE_KEYWORDS)]
        engineered_kept = [c for c in engineered_in_dataset if c in feature_cols]
        engineered_dropped = [c for c in engineered_in_dataset if c not in feature_cols]
        print(f"   → Contrôle features engineered : {len(engineered_kept)}/{len(engineered_in_dataset)} "
              f"colonnes (streak/momentum/quality_ema/delta_form/ema_long/r10_*) conservées comme features.")
        if engineered_dropped:
            print(f"     ⚠️ {len(engineered_dropped)} colonnes engineered exclues (probablement non numériques) :")
            for c in engineered_dropped[:20]:
                print(f"        - {c}")

    return feature_cols


def build_dmatrix(df, feature_cols, label_col=None):
    """Construit une DMatrix SANS remplacer les NaN par 0 : pour plusieurs de
    nos features (momentum_*, delta_form_*, r10_avg_opp_elo_beaten,
    quality_ema), 0 est une valeur réelle distincte de 'donnée manquante'.
    XGBoost gère nativement les NaN (missing=np.nan par défaut sur DMatrix)."""
    label = df[label_col] if label_col else None
    return xgb.DMatrix(df[feature_cols], label=label, missing=np.nan)


def train_and_eval_one_origin(df, cutoff, feature_cols):
    train = df[df["tourney_date"] < cutoff]
    test = df[(df["tourney_date"] >= cutoff) & (df["tourney_date"] <= TEST_WINDOW_END)]
    if len(train) < 500 or len(test) < 100:
        return None

    dtrain = build_dmatrix(train, feature_cols, "target_player1_won")
    dtest = build_dmatrix(test, feature_cols, "target_player1_won")
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
        # La baseline Elo (régression logistique sur 2 features) reste sur
        # fillna(0) : ici 0 pour un ÉCART d'Elo entre deux joueurs (diff=0)
        # a un sens raisonnable par défaut (aucun avantage), contrairement
        # aux features engineered ci-dessus où 0 collisionne avec un vrai 0.
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

    # --- Baseline "classement ATP brut" (principe #5 du README) ---------
    # Miroir exact de la baseline Elo : régression logistique sur 2 features
    # (écart de rang, écart de points ATP). Rang manquant (non classé) ->
    # RANK_UNRANKED_FALLBACK plutôt que 0, car 0 serait interprété comme
    # "meilleur rang possible" (rang #0), soit l'inverse de la réalité.
    # Points manquants -> 0, qui est ici la valeur réelle d'un joueur sans
    # point ATP (contrairement au rang, l'absence de points ATP vaut
    # littéralement 0 point).
    rank_metrics = None
    needed_rank = ["player1_rank", "player2_rank", "player1_rank_points", "player2_rank_points"]
    if all(c in df.columns for c in needed_rank):
        train_rank_p1 = train["player1_rank"].fillna(RANK_UNRANKED_FALLBACK)
        train_rank_p2 = train["player2_rank"].fillna(RANK_UNRANKED_FALLBACK)
        test_rank_p1 = test["player1_rank"].fillna(RANK_UNRANKED_FALLBACK)
        test_rank_p2 = test["player2_rank"].fillna(RANK_UNRANKED_FALLBACK)

        # Signe cohérent avec l'Elo : positif = player1 mieux classé/avantagé.
        # Rang : un rang NUMÉRIQUEMENT plus petit est MEILLEUR, donc on
        # inverse (player2_rank - player1_rank) pour que "positif" signifie
        # bien "player1 avantagé", comme pour elo_diff.
        train_rank_diff = (train_rank_p2 - train_rank_p1)
        test_rank_diff = (test_rank_p2 - test_rank_p1)
        train_points_diff = (train["player1_rank_points"] - train["player2_rank_points"]).fillna(0)
        test_points_diff = (test["player1_rank_points"] - test["player2_rank_points"]).fillna(0)

        X_train_rank = np.column_stack([train_rank_diff, train_points_diff])
        X_test_rank = np.column_stack([test_rank_diff, test_points_diff])
        clf_rank = LogisticRegression()
        clf_rank.fit(X_train_rank, train["target_player1_won"])
        proba_rank = clf_rank.predict_proba(X_test_rank)[:, 1]
        rank_metrics = {
            "accuracy": accuracy_score(y_test, proba_rank > 0.5),
            "log_loss": log_loss(y_test, proba_rank),
            "brier": brier_score_loss(y_test, proba_rank),
        }

    return {
        "cutoff": cutoff, "n_train": len(train), "n_test": len(test),
        "full": full_metrics, "elo": elo_metrics, "rank": rank_metrics,
    }


def bootstrap_ci_on_diffs(diffs, n_boot=5000, seed=0):
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


def save_feature_importance(model, feature_cols, output_dir):
    """Sauvegarde la table COMPLÈTE d'importance (plus de head(50)), et une
    liste séparée des features candidates jamais utilisées dans un split
    (gain nul -> absentes de model.get_score()). Avant, ces deux causes de
    'feature absente du tableau' (troncature à 50 vs gain réellement nul)
    étaient indiscernables."""
    gain = model.get_score(importance_type="gain")
    imp_table = pd.Series(gain).sort_values(ascending=False).rename("gain").to_frame()
    imp_table.to_csv(output_dir / "feature_importance_last_origin_full.csv")

    never_used = sorted(set(feature_cols) - set(gain.keys()))
    with open(output_dir / "feature_importance_never_used.txt", "w") as f:
        f.write(f"{len(never_used)} features candidates JAMAIS utilisées dans un split "
                f"(gain nul) sur cette origine :\n")
        for c in never_used:
            f.write(f"  {c}\n")

    engineered_never_used = [c for c in never_used if any(k in c for k in ENGINEERED_FEATURE_KEYWORDS)]
    print(f"\n📄 Importance COMPLÈTE ({len(imp_table)} features utilisées) sauvegardée dans "
          f"{output_dir / 'feature_importance_last_origin_full.csv'}")
    print(f"📄 {len(never_used)} features jamais utilisées listées dans "
          f"{output_dir / 'feature_importance_never_used.txt'} "
          f"(dont {len(engineered_never_used)} parmi nos features engineered)")
    if engineered_never_used:
        print("   Features engineered à gain nul sur cette origine :")
        for c in engineered_never_used[:20]:
            print(f"     - {c}")


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

    feature_cols = get_feature_cols(df, verbose=True)
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
        acc_rank = res["rank"]["accuracy"] if res["rank"] else float("nan")
        print(f"     n_train={res['n_train']}, n_test={res['n_test']}, "
              f"accuracy modèle={acc_full:.4f}, accuracy Elo={acc_elo:.4f}, "
              f"accuracy Rang ATP={acc_rank:.4f}, "
              f"gain vs Elo={acc_full - acc_elo:+.4f}, gain vs Rang={acc_full - acc_rank:+.4f}")

    if not per_origin_results:
        print("✗ Aucune origine n'a produit de résultat exploitable.")
        return

    print("\n=== Stabilité du gain (modèle complet - Elo) à travers les origines ===")
    acc_diffs = [r["full"]["accuracy"] - r["elo"]["accuracy"] for r in per_origin_results if r["elo"]]
    logloss_diffs = [r["elo"]["log_loss"] - r["full"]["log_loss"] for r in per_origin_results if r["elo"]]

    acc_mean, acc_lo, acc_hi = bootstrap_ci_on_diffs(acc_diffs)
    ll_mean, ll_lo, ll_hi = bootstrap_ci_on_diffs(logloss_diffs)

    print(f"Gain d'accuracy moyen (modèle - Elo) : {acc_mean:+.4f}  |  IC bootstrap 95% : [{acc_lo:+.4f} ; {acc_hi:+.4f}]")
    print(f"Gain de log-loss moyen (Elo - modèle, positif = mieux) : {ll_mean:+.4f}  |  IC bootstrap 95% : [{ll_lo:+.4f} ; {ll_hi:+.4f}]")
    if acc_lo > 0:
        print("🟢 Le gain d'accuracy du modèle complet sur Elo semble robuste à travers les périodes testées.")
    else:
        print("🟡 L'IC du gain d'accuracy inclut 0 : le gain n'est pas garanti stable sur toutes les périodes — "
              "regarder le détail par origine ci-dessus avant de conclure.")

    print("\n=== Stabilité du gain (modèle complet - Classement ATP brut) à travers les origines ===")
    acc_diffs_rank = [r["full"]["accuracy"] - r["rank"]["accuracy"] for r in per_origin_results if r["rank"]]
    logloss_diffs_rank = [r["rank"]["log_loss"] - r["full"]["log_loss"] for r in per_origin_results if r["rank"]]

    acc_mean_rank, acc_lo_rank, acc_hi_rank = bootstrap_ci_on_diffs(acc_diffs_rank)
    ll_mean_rank, ll_lo_rank, ll_hi_rank = bootstrap_ci_on_diffs(logloss_diffs_rank)

    print(f"Gain d'accuracy moyen (modèle - Rang ATP) : {acc_mean_rank:+.4f}  |  IC bootstrap 95% : "
          f"[{acc_lo_rank:+.4f} ; {acc_hi_rank:+.4f}]")
    print(f"Gain de log-loss moyen (Rang ATP - modèle, positif = mieux) : {ll_mean_rank:+.4f}  |  IC bootstrap 95% : "
          f"[{ll_lo_rank:+.4f} ; {ll_hi_rank:+.4f}]")
    if acc_lo_rank > 0:
        print("🟢 Le gain d'accuracy du modèle complet sur le rang ATP brut semble robuste à travers les périodes testées.")
    else:
        print("🟡 L'IC du gain d'accuracy vs rang ATP inclut 0 : le gain n'est pas garanti stable sur toutes les périodes.")

    # Comparaison directe des deux baselines entre elles : le rang ATP
    # brut est-il seulement redondant avec l'Elo, ou apporte-t-il quelque
    # chose de différent ? (informatif, pas utilisé pour juger le modèle)
    elo_vs_rank_acc = [r["elo"]["accuracy"] - r["rank"]["accuracy"] for r in per_origin_results if r["elo"] and r["rank"]]
    if elo_vs_rank_acc:
        evr_mean, evr_lo, evr_hi = bootstrap_ci_on_diffs(elo_vs_rank_acc)
        print(f"\nPour référence — écart d'accuracy Elo vs Rang ATP brut (indépendamment du modèle complet) : "
              f"{evr_mean:+.4f}  |  IC bootstrap 95% : [{evr_lo:+.4f} ; {evr_hi:+.4f}]")

    # Importance des features sur le dernier modèle entraîné (origine la plus récente)
    last_train = df[df["tourney_date"] < TEST_ORIGINS[-1]]
    last_test = df[(df["tourney_date"] >= TEST_ORIGINS[-1]) & (df["tourney_date"] <= TEST_WINDOW_END)]
    if len(last_train) > 500 and len(last_test) > 100:
        dtrain = build_dmatrix(last_train, feature_cols, "target_player1_won")
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 5,
                  "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}
        model = xgb.train(params, dtrain, num_boost_round=200)
        save_feature_importance(model, feature_cols, OUTPUTS_DIR)

    summary_rows = []
    for r in per_origin_results:
        row = {"cutoff": r["cutoff"], "n_train": r["n_train"], "n_test": r["n_test"],
               "accuracy_full": r["full"]["accuracy"], "log_loss_full": r["full"]["log_loss"]}
        if r["elo"]:
            row.update({"accuracy_elo": r["elo"]["accuracy"], "log_loss_elo": r["elo"]["log_loss"]})
        if r["rank"]:
            row.update({"accuracy_rank_atp": r["rank"]["accuracy"], "log_loss_rank_atp": r["rank"]["log_loss"]})
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUTS_DIR / "walkforward_summary.csv", index=False)

    with open(OUTPUTS_DIR / "walkforward_summary.txt", "w") as f:
        f.write(summary_df.to_string(index=False))
        f.write(f"\n\n--- vs baseline Elo ---\n")
        f.write(f"Gain accuracy moyen : {acc_mean:+.4f}, IC95%: [{acc_lo:+.4f}; {acc_hi:+.4f}]\n")
        f.write(f"Gain log-loss moyen : {ll_mean:+.4f}, IC95%: [{ll_lo:+.4f}; {ll_hi:+.4f}]\n")
        f.write(f"\n--- vs baseline Classement ATP brut ---\n")
        f.write(f"Gain accuracy moyen : {acc_mean_rank:+.4f}, IC95%: [{acc_lo_rank:+.4f}; {acc_hi_rank:+.4f}]\n")
        f.write(f"Gain log-loss moyen : {ll_mean_rank:+.4f}, IC95%: [{ll_lo_rank:+.4f}; {ll_hi_rank:+.4f}]\n")

    print(f"\n✅ Résumé écrit dans {OUTPUTS_DIR / 'walkforward_summary.txt'} "
          f"et {OUTPUTS_DIR / 'walkforward_summary.csv'}")


if __name__ == "__main__":
    main()