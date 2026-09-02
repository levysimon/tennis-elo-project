# ATP Tennis Match Prediction

A fully reproducible pipeline that predicts ATP match winners from four feature families (surface-specific Elo, serve/return efficiency, fatigue/recent form, head-to-head confrontation), and rigorously tests whether that predictive edge translates into a betting edge against closing odds.

The model beats an Elo-only baseline by a small but statistically robust margin (+1.6 accuracy points, 95% CI excludes zero, stable across 4 independent test periods 2022–2025). It finds **no evidence of a profitable betting edge** against Pinnacle closing odds on this dataset (holdout ROI ≈ −3.7%, p = 1.00). Both results are reported as-is - this project is a methodology exercise, not a betting system.



## Motivation

ATP rankings are a noisy, slow-reacting signal of player strength. This project builds a more reactive, surface-specific alternative (Elo per surface) and combines it with recent-form, fatigue, and head-to-head features, then asks two separate questions:

1. **Does this improve on Elo alone at predicting the match winner?**
2. **Does any resulting edge survive contact with a real, efficient betting market (Pinnacle)?**

The academic literature and professional bettors broadly agree that even strong published models cap out around 65-70% winner prediction accuracy, and that beating a sharp bookmaker consistently is a much harder problem than being "merely accurate". This project treats that as a hypothesis to test empirically on its own data, not an assumption.

## Data

**Primary source:** [TennisMyLife Match Database](https://stats.tennismylife.org/tennis-match-database), a database derived from Jeff Sackmann / Tennis Abstract's work, with official ATP player IDs, covering 1986–2026.

**Betting odds:** historical odds files (Pinnacle `PSW`/`PSL`, Bet365 `B365W`/`B365L`, plus `Max`/`Avg` where available) from [tennis-data.co.uk](http://tennis-data.co.uk/data.php). 


## Methodology principles

These rules exist specifically to avoid the most common failure mode in this kind of project: **data leakage and retrospective selection bias that inflate apparent performance.**

1. **Strict chronological order.** Every feature (Elo, rolling average, H2H) is computed only from matches *prior* to the predicted match. No future information ever leaks backward.
2. **Temporal splits only, never random.** Training and evaluation periods are always separated by date, never shuffled.
3. **No features derived from the outcome.** A feature like "rank difference" is computed from pre-match ranks, never reconstructed from winner/loser labels.
4. **A frozen selection/holdout split for any strategy parameter.** Betting filter parameters (confidence threshold, odds cap, sorting metric) are chosen on a "selection" period only, then applied ***once, unchanged***, to an independent "holdout" period. The holdout number — not the selection-period number — is the one reported as the performance estimate.
5. **Always benchmark against a simple baseline.** Elo alone (and raw ATP ranking) is the reference point for judging whether additional features actually add value.
6. **Statistical uncertainty is always reported.** Every ROI or accuracy-gain figure comes with a bootstrap confidence interval and/or permutation-test p-value, not a bare point estimate.

## Pipeline

1. **Collection** — download yearly ATP CSVs (2000–2026), verify row counts / expected columns, document missing-value rates per column/year.
2. **Cleaning** — concatenate, sort by date, normalize player IDs, ?? filter to hard-court matches?? vraiment ? 
3. **Merge maths and odds** — one row per match, symmetrized as player1/player2 to avoid systematically encoding the winner first; odds merged by player-pair + year.
4. **Elo** — global and surface-specific Elo, computed match-by-match in chronological order.
5. **Serve/return features** — 1st/2nd serve win rates, break-point conversion/save rates, rolling averages (last 10 matches, last 12 months, exponential mobile average).
6. **Fatigue/form** — sets and minutes played in the trailing 30 days and with exponential mobile average + number of rest days since the last match.
7. **Head-to-head confrontation** — cumulative, chronologically updated, both overall by surface.
8. **Building dataset**
9. **Modeling** — XGBoost classifier vs. a logistic-regression-on-Elo-difference baseline, evaluated with a multi-origin walk-forward backtest (2022, 2023, 2024, 2025 → present).
10. **Betting evaluation** — no-vig market probability, confidence ratio and absolute-edge sizing, flat betting (1$ on each chosen bet), majority-vote ensemble across 7 shifted training windows, frozen selection/holdout split, bootstrap CI + permutation test.
11. **XGBoost evaluation** - against Elo baseline, comparison on multiple years.

## Results
### Predictive performance vs. Elo baseline

Multi-origin walk-forward backtest (train on all data before the cutoff, test on everything after):

| Cutoff | n_train | n_test | Accuracy (full model) | Log-loss (full model) | Accuracy (Elo baseline) | Log-loss (Elo baseline) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 2022-01-01 | 39,170 | 9,859 | 0.6710 | 0.5857 | 0.6384 | 0.6261 |
| 2023-01-01 | 41,673 | 7,356 | 0.6725 | 0.5891 | 0.6346 | 0.6305 |
| 2024-01-01 | 44,107 | 4,922 | 0.6699 | 0.5923 | 0.6335 | 0.6296 |
| 2025-01-01 | 46,652 | 2,377 | 0.6676 | 0.5964 | 0.6294 | 0.6356 |

**Mean accuracy gain (full model − Elo): +3.63 points, bootstrap 95% CI [+3.39 ; +3.81]** — excludes zero, and the gain is consistent across all four independent test periods rather than driven by a single lucky split.

**Mean log-loss gain: +0.0396, 95% CI [+0.0381 ; +0.0409]** — same conclusion.

This sits within the ~65–70% ceiling widely reported in the tennis-prediction literature, and confirms that the additional feature families (serve/return, fatigue, H2H) carry real, reproducible information beyond Elo alone.

### Betting ROI evaluation

Walk-forward strategy (flat betting, frozen selection/holdout split):

| | Selection period | Holdout (frozen strategy, never seen during selection) |
|---|---|---|
| Bets | 216 (best grid combo: `edge_abs` sort, top 50%, odds ≤ 3) | — |
| ROI | +7.47% (bootstrap mean) | **−0.12%** |
| Bootstrap 95% CI | [−4.13% ; +18.85%] | **[−8.19% ; +8.43%]** |
| Permutation p-value | — | **1.0000** |

Independently, a second evaluation script (chronological A/B split, threshold chosen on A only, top 30% threshold with $n=189$) yielded a holdout ROI of **+15.92%**, 95% CI **[−3.25% ; +35.31%]**, and p = **0.7248**. 

While specific subsets show high returns, overall statistical tests (p-values around 0.72 to 1.00 and confidence intervals spanning zero) indicate insufficient evidence of a robust, generalized edge out-of-sample once accounting for variance.

ROI by confidence decile (descriptive, full sample):

| Decile | n bets | Avg. confidence | ROI |
|---|---|---|---|
| 0–10% (highest confidence) | 126 | 1.95 | +50.33% |
| 10–20% | 126 | 1.30 | +0.67% |
| 20–30% | 126 | 1.19 | +4.48% |
| 30–40% | 126 | 1.13 | −4.55% |
| 40–50% | 126 | 1.09 | −3.11% |
| 50–60% | 126 | 1.05 | +12.83% |
| 60–70% | 126 | 1.02 | −0.22% |
| 70–80% | 126 | 0.98 | +0.07% |
| 80–90% | 126 | 0.94 | +4.24% |
| 90–100% (lowest confidence) | 126 | 0.85 | −8.49% |


**Conclusion: no statistically defensible evidence of a profitable edge against Pinnacle closing odds on this dataset**, despite a real and stable predictive edge over Elo. This is consistent with market efficiency: a bookmaker with access to the same or better information prices it in before the bet is placed. See [Possible extensions](#possible-extensions) for what would be worth testing next (opening odds, softer books, thinner markets).

### Feature importance

Top drivers by XGBoost gain (most recent walk-forward origin):

| Feature | Gain |
|---|---|
| `player2_seed` | 121.6 |
| `player1_seed` | 92.2 |
| `player2_elo_pre` | 63.7 |
| `player1_elo_pre` | 45.1 |
| `h2h_smoothed_diff` | 41.1 |
| `player1_surface_env_elo_pre` | 31.8 |
| `player1_surface_elo_pre` | 29.3 |
| `player2_surface_elo_pre` | 28.8 |
| `player1_rank` | 27.9 |
| `h2h_surface_smoothed_diff` | 25.9 |

(full table in `outputs/feature_importance_last_origin.csv`)

Elo-family features and seeding dominate, which is expected and reassuring — it means the model isn't leaning on a spurious signal to hit its accuracy numbers.


## Limitations

- Holdout sample sizes for the betting evaluation are modest (100–400 bets), giving wide confidence intervals. The conclusion is "no evidence of an edge with the strategies tested," not "no edge could possibly exist under any strategy."
- The betting evaluation only uses Pinnacle closing odds. Opening odds, other bookmakers, and other tours/tiers were not tested.
- Injury history, weather, etc... are not currently incorporated as features.

## Possible extensions
- **Please, (someone) do it for the WTA circuit! (obivous)**

---

*This project is exploratory. It is not a betting system, and nothing here should be read as gambling advice. Be smart and don't bet :)*
