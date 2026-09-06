#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 3 — Flat Betting (Mise fixe) et suppression de Kelly.

CHANGEMENTS PAR RAPPORT A LA V2 :
1. Suppression totale du critère de Kelly. La mise ("stake") est systématiquement de 1.0 
   dès que le modèle détecte une espérance de gain positive (EV > 0).
2. Simplification des fonctions de calcul de rentabilité (profitComputation, bootstrap, etc.) 
   qui utilisent désormais nativement cette mise fixe.

CHANGEMENTS PAR RAPPORT A LA V3 (ce fichier) :
3. Remplacement de l'exclusion par PRÉFIXE (startswith) des colonnes de cotes par une
   exclusion par NOM EXACT. L'ancienne approche (ODDS_LEAKAGE_PREFIXES avec des chaînes
   courtes comme 'player1_ps', 'player1_avg', 'player1_max') fonctionnait correctement
   avec les colonnes actuelles, MAIS était fragile : toute future feature nommée
   'player1_avg_XXX' ou 'player1_max_XXX' aurait été silencieusement exclue du modèle
   sans aucun avertissement, alors qu'elle n'a rien à voir avec les cotes de bookmaker.
   La liste exacte ci-dessous élimine ce risque, et un contrôle de cohérence est imprimé
   au lancement pour vérifier explicitement que les features engineered (streak,
   momentum, quality_ema, delta_form, ema_long, r10_*) sont bien conservées.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "test roi"

DATASET_FILE = PROCESSED_DIR / "atp_modeling_dataset.csv"

# --- Fenêtres temporelles ------------------------------------------------
DATA_START = "2000-01-01"
SELECTION_TEST_START = "2017-01-01"
HOLDOUT_START = "2020-01-01"

# --- Colonnes de cotes de bookmaker : EXCLUSION PAR NOM EXACT -------------
# Ancien mécanisme (prefix-based, fragile) :
#   ODDS_LEAKAGE_PREFIXES = ('player1_b365', 'player1_ps', 'player1_ex',
#                            'player1_max', 'player1_avg', 'player2_b365',
#                            'player2_ps', 'player2_ex', 'player2_max', 'player2_avg')
# Problème : 'player1_ps'.startswith(...) matche n'importe quelle future colonne
# commençant par ces mêmes lettres (ex: 'player1_avg_opp_elo_beaten' aurait pu être
# piégée si elle avait porté ce nom). On liste maintenant les colonnes UNE PAR UNE.
BOOKMAKER_SUFFIXES = ('b365', 'ps', 'ex', 'max', 'avg')
ODDS_LEAKAGE_EXACT_COLS = {
    f"player{n}_{suffix}" for n in (1, 2) for suffix in BOOKMAKER_SUFFIXES
}
# Autres colonnes exactes à exclure (fuite d'information / non-features)
OTHER_LEAKAGE_EXACT_COLS = {'player1_elo_win_prob'}

ALL_EXACT_EXCLUDED_COLS = ODDS_LEAKAGE_EXACT_COLS | OTHER_LEAKAGE_EXACT_COLS

# Mots-clés utilisés uniquement pour le contrôle de cohérence affiché au lancement
# (vérifier visuellement que nos features engineered ne sont jamais exclues par erreur)
ENGINEERED_FEATURE_KEYWORDS = (
    "streak", "quality_ema", "momentum_", "delta_form_", "ema_long_",
    "r10_avg_opp_elo_beaten", "r10_upset_win_pct", "r10_bad_loss_pct",
)


def xgbModelBinary(xtrain, ytrain, xval, yval, p):
    dtrain = xgb.DMatrix(xtrain, label=ytrain)
    dval = xgb.DMatrix(xval, label=yval)
    eval_set = [(dtrain, "train"), (dval, "eval")]
    params = {
        'eval_metric': "logloss",
        'objective': "binary:logistic",
        'subsample': 0.8,
        'min_child_weight': p.get('min_child_weight', 1),
        'alpha': p.get('alpha', 1),
        'lambda': p.get('lambda', 1),
        'max_depth': int(p.get('max_depth', 5)),
        'gamma': p.get('gamma', 0),
        'eta': p.get('learning_rate', 0.1),
        'colsample_bytree': p.get('colsample_bytree', 0.8)
    }
    model = xgb.train(
        params, dtrain,
        num_boost_round=int(p.get('num_rounds', 200)),
        evals=eval_set,
        early_stopping_rounds=int(p.get('early_stop', 10)),
        verbose_eval=False
    )
    return model


def get_candidate_feature_cols(columns) -> list[str]:
    """Détermine les colonnes candidates à devenir des features, en excluant :
    - les colonnes de métadonnées/résultat (drop_cols)
    - les colonnes de cotes de bookmaker et player1_elo_win_prob (nom EXACT, pas préfixe)
    - les colonnes de noms de joueurs (player1_name / player2_name)
    Le typage numérique final est décidé ensuite par select_dtypes (voir appelant).
    """
    drop_cols = {
        'target_player1_won', 'tourney_date', 'tourney_id', 'match_num',
        'dataset_split', 'minutes', 'l_1stIn', 'w_1stIn', 'l_svpt', 'w_svpt'
    }
    return [
        c for c in columns
        if c not in drop_cols
        and c not in ALL_EXACT_EXCLUDED_COLS
        and c not in ('player1_name', 'player2_name')
    ]


def log_feature_selection_sanity_check(df_train_columns, feature_cols) -> None:
    """Affiche un contrôle de cohérence : combien de nos features engineered
    (streak, momentum, quality_ema, delta_form, ema_long, r10_*) sont bien
    présentes dans feature_cols. Sert à détecter immédiatement toute
    régression future (ex: renommage de colonne, nouveau filtre trop large)."""
    engineered_in_dataset = [
        c for c in df_train_columns
        if any(k in c for k in ENGINEERED_FEATURE_KEYWORDS)
    ]
    engineered_kept = [c for c in engineered_in_dataset if c in feature_cols]
    engineered_dropped = [c for c in engineered_in_dataset if c not in feature_cols]

    print(f"→ Contrôle features engineered : {len(engineered_kept)}/{len(engineered_in_dataset)} "
          f"colonnes (streak/momentum/quality_ema/delta_form/ema_long/r10_*) conservées comme features.")
    if engineered_dropped:
        print(f"  ⚠️ ATTENTION — {len(engineered_dropped)} colonnes engineered exclues par erreur "
              f"(probablement non numériques après lecture CSV) :")
        for c in engineered_dropped[:20]:
            print(f"     - {c}")


def assessStrategySingleModel(df_train, df_val, df_test, xgb_params, model_name="1", verbose_check=False):
    if len(df_test) == 0 or len(df_train) == 0:
        return None

    candidate_cols = get_candidate_feature_cols(df_train.columns)
    feature_cols = df_train[candidate_cols].select_dtypes(include=[np.number, bool, 'category']).columns.tolist()

    if verbose_check:
        log_feature_selection_sanity_check(df_train.columns, feature_cols)

    xtrain = df_train[feature_cols]
    ytrain = df_train['target_player1_won']
    xval = df_val[feature_cols]
    yval = df_val['target_player1_won']
    xtest = df_test[feature_cols]

    model = xgbModelBinary(xtrain, ytrain, xval, yval, xgb_params)

    pred_test_p1 = model.predict(xgb.DMatrix(xtest))
    pred_test_p2 = 1.0 - pred_test_p1

    odds_p1 = pd.to_numeric(df_test['player1_ps'], errors='coerce').fillna(0).values
    odds_p2 = pd.to_numeric(df_test['player2_ps'], errors='coerce').fillna(0).values

    raw_p1 = np.divide(1.0, odds_p1, out=np.zeros_like(odds_p1, dtype=float), where=odds_p1 > 0)
    raw_p2 = np.divide(1.0, odds_p2, out=np.zeros_like(odds_p2, dtype=float), where=odds_p2 > 0)
    margin_sum = raw_p1 + raw_p2

    implied_p1_novig = np.divide(raw_p1, margin_sum, out=np.zeros_like(raw_p1, dtype=float), where=margin_sum > 0)
    implied_p2_novig = np.divide(raw_p2, margin_sum, out=np.zeros_like(raw_p2, dtype=float), where=margin_sum > 0)

    right = (pred_test_p1 > pred_test_p2).astype(int)

    pmodel_chosen = np.where(right == 1, pred_test_p1, pred_test_p2)
    pmarket_novig_chosen = np.where(right == 1, implied_p1_novig, implied_p2_novig)
    chosen_odds = np.where(right == 1, odds_p1, odds_p2)

    confidences = np.divide(
        pmodel_chosen, pmarket_novig_chosen,
        out=np.zeros_like(pmodel_chosen, dtype=float), where=pmarket_novig_chosen > 0
    )
    edge_abs_novig = pmodel_chosen - pmarket_novig_chosen

    # --- SUPPRESSION DE KELLY -> FLAT BETTING ---
    ev = (pmodel_chosen * chosen_odds) - 1.0
    # On met systématiquement 1 si l'espérance est positive, 0 sinon
    stake = np.where(ev > 0, 1.0, 0.0)

    actual_p1_won = df_test['target_player1_won'].values
    bet_won = ((right == 1) & (actual_p1_won == 1)) | ((right == 0) & (actual_p1_won == 0))

    result_df = pd.DataFrame({
        "match_idx": df_test.index,
        "tourney_date": df_test['tourney_date'].values,
        f"win_{model_name}": right,
        f"conf_{model_name}": confidences,
        f"edge_abs_{model_name}": edge_abs_novig,
        f"stake_{model_name}": stake,
        f"odds_{model_name}": chosen_odds,
        f"bet_won_{model_name}": bet_won.astype(int),
        f"pmodel_{model_name}": pmodel_chosen,
        f"pmarket_novig_{model_name}": pmarket_novig_chosen,
    })
    return result_df


def run_majority_voting_ensemble(df_full, test_start_idx, train_window_days=730, val_window_days=180,
                                  test_window_days=90, xgb_params={}, verbose_check=False):
    df_full = df_full.sort_values('tourney_date').reset_index(drop=True)
    offsets = [0, -20, 20, -50, 50, -90, 90]
    sub_dfs_conf = []

    for i, offset in enumerate(offsets):
        model_name = str(i + 1)
        test_end_idx = min(test_start_idx + test_window_days, len(df_full))
        val_end_idx = test_start_idx
        val_start_idx = max(0, val_end_idx - int(val_window_days * (len(df_full) / 365.0 / 5.0)))
        n_train_rows = 10000 + offset * 10
        train_start_idx = max(0, val_start_idx - n_train_rows)

        df_train = df_full.iloc[train_start_idx:val_start_idx]
        df_val = df_full.iloc[val_start_idx:val_end_idx]
        df_test = df_full.iloc[test_start_idx:test_end_idx]

        if len(df_test) == 0 or len(df_train) == 0:
            continue

        # Contrôle de cohérence des features affiché une seule fois (premier
        # sous-modèle du tout premier appel), pour ne pas spammer les logs
        # tout en garantissant une vérification effective à chaque run.
        res = assessStrategySingleModel(
            df_train, df_val, df_test, xgb_params, model_name=model_name,
            verbose_check=(verbose_check and i == 0)
        )
        if res is not None:
            sub_dfs_conf.append(res.set_index('match_idx'))

    if not sub_dfs_conf:
        return None

    merged = pd.concat(sub_dfs_conf, axis=1)
    win_cols = [c for c in merged.columns if c.startswith('win_')]
    conf_cols = [c for c in merged.columns if c.startswith('conf_')]
    edge_cols = [c for c in merged.columns if c.startswith('edge_abs_')]
    stake_cols = [c for c in merged.columns if c.startswith('stake_')]
    odds_cols = [c for c in merged.columns if c.startswith('odds_')]
    won_cols = [c for c in merged.columns if c.startswith('bet_won_')]

    wins_sum = merged[win_cols].sum(axis=1)
    majority_win = (wins_sum >= 4).astype(int)

    final_conf = merged[conf_cols].mean(axis=1)
    final_edge_abs = merged[edge_cols].mean(axis=1)
    final_stake = merged[stake_cols].mean(axis=1)

    if len(odds_cols) == 0 or len(won_cols) == 0:
        return None

    final_odds = merged[odds_cols[0]]
    final_won = merged[won_cols[0]]

    final_date = merged.loc[:, merged.columns.get_loc('tourney_date')] if merged.columns.tolist().count('tourney_date') == 1 else merged.iloc[:, [c == 'tourney_date' for c in merged.columns]].iloc[:, 0]

    final_df = pd.DataFrame({
        "match_idx": merged.index,
        "tourney_date": final_date.values if hasattr(final_date, 'values') else final_date,
        "win_majority": majority_win,
        "confidence": final_conf,
        "edge_abs": final_edge_abs,
        "stake": final_stake,
        "PSW": final_odds,
        "bet_won": final_won
    })
    return final_df


def profitComputation(df):
    """Calcul du ROI basé sur la colonne stake (mise fixe de 1.0)"""
    if len(df) == 0:
        return np.nan
    total_staked = df['stake'].sum()
    if total_staked == 0:
        return np.nan
    returns = (df['stake'] * df['PSW'] * df['bet_won']).sum()
    return 100.0 * (returns - total_staked) / total_staked


def bootstrap_ci(df, n_boot=3000, seed=0, ci=(2.5, 97.5)):
    if len(df) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(df)
    idx_all = np.arange(n)
    rois = np.empty(n_boot)
    for i in range(n_boot):
        sample_idx = rng.choice(idx_all, size=n, replace=True)
        rois[i] = profitComputation(df.iloc[sample_idx])
    lo, hi = np.nanpercentile(rois, ci)
    return np.nanmean(rois), lo, hi


def permutation_pvalue(df, n_permutations=10000, seed=0):
    if len(df) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    odds = df["PSW"].values
    outcomes = df["bet_won"].values
    stakes = df['stake'].values 
    total_staked = stakes.sum()
    
    if total_staked == 0:
        return np.nan
        
    actual_roi = profitComputation(df)
    random_rois = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(outcomes)
        returns = np.sum(stakes * odds * shuffled)
        random_rois[i] = 100.0 * (returns - total_staked) / total_staked
    return np.sum(random_rois >= actual_roi) / n_permutations


def select_strategy_on_selection_period(sel_df, sort_cols=("edge_abs", "confidence"),
                                         percentages=(10, 20, 30, 40, 50),
                                         odds_max_grid=(2, 2.5, 3),
                                         min_bets=200):
    best = None
    for sort_col, pct, odds_max in product(sort_cols, percentages, odds_max_grid):
        pool = sel_df[(sel_df['PSW'] > 0) & (sel_df['PSW'] <= odds_max)]
        if len(pool) == 0:
            continue
        pool_sorted = pool.sort_values(sort_col, ascending=False).reset_index(drop=True)
        k = int(len(pool_sorted) * pct / 100)
        if k < min_bets:
            continue
        subset = pool_sorted.head(k)
        mean_roi, lo, hi = bootstrap_ci(subset)
        candidate = {
            "sort_col": sort_col, "percentage": pct, "odds_max": odds_max,
            "n_bets": k, "roi_mean_boot": mean_roi, "roi_ci_low": lo, "roi_ci_high": hi
        }
        if best is None or candidate["roi_ci_low"] > best["roi_ci_low"]:
            best = candidate
    return best


def apply_frozen_strategy(df, params):
    pool = df[(df['PSW'] > 0) & (df['PSW'] <= params['odds_max'])]
    pool_sorted = pool.sort_values(params['sort_col'], ascending=False).reset_index(drop=True)
    k = int(len(pool_sorted) * params['percentage'] / 100)
    return pool_sorted.head(k)


def main():
    print("→ Chargement du dataset symétrisé...")
    if not DATASET_FILE.exists():
        print(f"✗ Erreur : {DATASET_FILE} introuvable. Lance `build_dataset.py` d'abord.")
        return

    df = pd.read_csv(DATASET_FILE, low_memory=False)
    df['tourney_date'] = df['tourney_date'].astype(str)
    df = df[df['tourney_date'] >= DATA_START].reset_index(drop=True)
    print(f"→ {len(df)} matchs chargés depuis {DATA_START}.")

    xgb_params = {
        'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 5, 'gamma': 0.1,
        'colsample_bytree': 0.9, 'subsample': 0.7, 'lambda': 1, 'alpha': 1,
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'num_rounds': 80, 'early_stop': 10
    }

    test_start_indices = df[df['tourney_date'] >= SELECTION_TEST_START].index
    if len(test_start_indices) == 0:
        print(f"✗ Aucun match trouvé à partir de {SELECTION_TEST_START}.")
        return

    first_test_idx = test_start_indices[0]
    step = 1500
    all_test_results = []
    first_iteration = True
    for current_test_idx in range(first_test_idx, len(df), step):
        res = run_majority_voting_ensemble(
            df, test_start_idx=current_test_idx, xgb_params=xgb_params,
            verbose_check=first_iteration
        )
        first_iteration = False
        if res is not None:
            all_test_results.append(res)

    if not all_test_results:
        print("✗ La stratégie n'a généré aucun résultat.")
        return

    final_strategy_df = pd.concat(all_test_results, ignore_index=True)
    final_strategy_df['tourney_date'] = final_strategy_df['tourney_date'].astype(str)
    final_strategy_df = final_strategy_df.sort_values('tourney_date').reset_index(drop=True)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    # Renommé pour refléter la suppression de Kelly
    full_out_path = OUTPUTS_DIR / "sliding_confidence_novig_flat_full.csv"
    final_strategy_df.to_csv(full_out_path, index=False)
    print(f"→ Résultats bruts (toutes périodes, triés par date) sauvegardés dans {full_out_path}")

    # --- SPLIT SELECTION / HOLDOUT, GELE ------------------------------
    selection_df = final_strategy_df[final_strategy_df['tourney_date'] < HOLDOUT_START].copy()
    holdout_df = final_strategy_df[final_strategy_df['tourney_date'] >= HOLDOUT_START].copy()
    print(f"\n→ Période de sélection : {len(selection_df)} paris (< {HOLDOUT_START})")
    print(f"→ Période holdout (jamais vue pendant la sélection) : {len(holdout_df)} paris (>= {HOLDOUT_START})")

    if len(selection_df) < 200 or len(holdout_df) < 100:
        print("⚠️ Attention : échantillons petits, les conclusions ci-dessous seront peu robustes.")

    best_params = select_strategy_on_selection_period(selection_df)
    if best_params is None:
        print("✗ Aucune combinaison de paramètres n'a atteint le nombre minimum de paris sur la période de sélection.")
        return

    print("\n=== Paramètres retenus sur la période de SÉLECTION uniquement ===")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Application gelée, une seule fois, sur le holdout
    holdout_bets = apply_frozen_strategy(holdout_df, best_params)
    print(f"\n=== Application GELÉE de cette stratégie sur le HOLDOUT ({len(holdout_bets)} paris) ===")
    if len(holdout_bets) == 0:
        print("✗ Aucun pari sélectionné sur le holdout avec ces paramètres.")
        return

    holdout_roi = profitComputation(holdout_bets)
    boot_mean, boot_lo, boot_hi = bootstrap_ci(holdout_bets)
    p_value = permutation_pvalue(holdout_bets)

    print(f"ROI observé (Flat Betting) sur le holdout : {holdout_roi:.2f}%")
    print(f"Intervalle de confiance bootstrap 95% : [{boot_lo:.2f}% ; {boot_hi:.2f}%]")
    print(f"P-value (test de permutation) : {p_value:.4f}")
    if p_value < 0.05 and boot_lo > 0:
        print("🟢 Edge potentiellement réel : significatif ET intervalle de confiance > 0.")
    else:
        print("🔴 Pas de preuve suffisante d'un edge réel sur le holdout (voir p-value / IC).")

    holdout_bets.to_csv(OUTPUTS_DIR / "holdout_bets_frozen_strategy.csv", index=False)
    with open(OUTPUTS_DIR / "holdout_report.txt", "w") as f:
        f.write("Paramètres choisis sur la période de sélection (jamais sur le holdout) :\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nROI holdout (Flat Betting) : {holdout_roi:.2f}%\n")
        f.write(f"IC bootstrap 95% : [{boot_lo:.2f}% ; {boot_hi:.2f}%]\n")
        f.write(f"P-value permutation : {p_value:.4f}\n")
    print(f"\n✅ Rapport écrit dans {OUTPUTS_DIR / 'holdout_report.md'}")


if __name__ == "__main__":
    main()