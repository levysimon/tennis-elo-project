"""
build_features.py
==================
Étape 4 du projet : features de service/retour, lissées par moyenne glissante,
+ momentum/tendance et qualité des adversaires récents (Étape 4bis).

Pour chaque match, on dérive PAR JOUEUR :
  - srv_1st_pct       : % de points gagnés au 1er service
  - srv_2nd_pct       : % de points gagnés au 2e service
  - srv_bp_saved      : % de balles de break sauvées (au service)
  - ret_pts_pct       : % de points gagnés au retour (déduit des stats de
                        service de l'ADVERSAIRE)
  - ret_bp_conv       : % de balles de break converties (déduit du
                        bpFaced/bpSaved de l'adversaire au service)
  - hold_pct          : % de jeux de service conservés (approximation à
                        partir des balles de break converties par
                        l'adversaire, cf. docstring de compute_match_player_stats)
  - break_pct         : % des jeux de service adverses effectivement cassés
  - dominance_ratio   : (% points gagnés au retour) / (% points perdus au
                        service) — > 1.0 = joueur qui domine l'échange
  - pressure_rating   : srv_bp_saved + ret_bp_conv, proxy de solidité sur
                        les points de break (approximation ; ne capture pas
                        spécifiquement les tie-breaks, cf. limites plus bas)

Ces métriques sont ensuite lissées de TROIS façons :
  1. Moyenne glissante sur les 10 DERNIERS MATCHS JOUÉS, toutes surfaces
     confondues ("forme récente" générale).
  2. Moyenne glissante sur les 12 DERNIERS MOIS, UNIQUEMENT sur dur
     ("forme récente" spécifique à la surface qui nous intéresse).
  3. Moyenne mobile exponentielle (EMA) COURTE, pondérée par la récence en
     JOURS (demi-vie = EMA_HALF_LIFE_DAYS) plutôt que par un nombre fixe de
     matchs ou une fenêtre calendaire dure — absorbe mieux les pauses
     saisonnières du calendrier tennis qu'une fenêtre à bornes fixes.

Étape 4bis — Momentum et qualité des adversaires (AJOUT) :
  4. EMA LONGUE (demi-vie = EMA_LONG_HALF_LIFE_DAYS, 120j par défaut) sur
     les mêmes métriques que l'EMA courte. Sert de référence "niveau de fond"
     pour détecter une tendance : si la forme courte s'écarte nettement de
     la forme longue, le joueur est en progression ou en déclin.
  5. `delta_form_*` = ema_{metric} (courte) - ema_long_{metric}. Positif =
     le joueur performe mieux que sa moyenne longue durée en ce moment
     (dynamique montante) ; négatif = dynamique descendante.
  6. `momentum_*` (dominance_ratio, hold_pct) = différence entre la moyenne
     de la 2e moitié et celle de la 1re moitié des ROLLING_WINDOW_N derniers
     matchs. Capture la pente à l'intérieur même de la fenêtre r10, distincte
     du delta_form qui compare deux fenêtres temporelles différentes.
  7. `streak` : série de victoires/défaites consécutives, signée (+3 = 3
     victoires de suite, -2 = 2 défaites de suite). Remise à 1/-1 dès le
     premier résultat de signe opposé.
  8. Qualité des adversaires sur les ROLLING_WINDOW_N derniers matchs :
       - `r10_avg_opp_elo_beaten` : Elo (surface si disponible, sinon
         global) moyen des adversaires BATTUS dans la fenêtre.
       - `r10_upset_win_pct` : % des victoires de la fenêtre obtenues contre
         un adversaire mieux coté (Elo pré-match supérieur au sien).
       - `r10_bad_loss_pct` : % des défaites de la fenêtre concédées contre
         un adversaire moins bien coté (signal souvent plus informatif
         qu'une simple série de victoires faciles).
       - `quality_ema` : EMA (même demi-vie que l'EMA courte) de
         (elo_adversaire - elo_propre) calculée UNIQUEMENT sur les matchs
         gagnés — mesure la qualité pondérée par la récence des victoires
         récentes, indépendamment du nombre de matchs joués.
  Ces features nécessitent l'Elo pré-match des DEUX joueurs (colonnes issues
  de build_elo.py). Cf. `get_elo_pre()` pour la logique de repli si l'Elo par
  surface n'est pas disponible.

Gestion de l'historique insuffisant :
Si un joueur a moins de MIN_HISTORY_FOR_AVERAGE matchs valides, la moyenne
(quel que soit le mode) n'est PAS calculée (case vide) plutôt que de
produire une estimation bruitée sur 1 ou 2 matchs. Le nombre de matchs
réellement disponibles est toujours indiqué à côté (colonnes `*_n`).
Le momentum (calcul de pente sur 2 demi-fenêtres) exige 2x ce seuil
(MIN_HISTORY_FOR_MOMENTUM) pour avoir au moins MIN_HISTORY_FOR_AVERAGE
matchs valides de CHAQUE côté de la coupure.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --input atp_matches_with_elo.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" /"serve features"

DEFAULT_INPUT_FILE = PROCESSED_DIR / "atp_matches_with_elo.csv"
OUTPUT_FILE = PROCESSED_DIR / "atp_matches_with_features.csv"
OUTPUT_REPORT_FILE = OUTPUTS_DIR / "build_features_report.md"

ROLLING_WINDOW_N = 10          # nombre de matchs pour la moyenne "forme récente" toutes surfaces
ROLLING_WINDOW_DAYS_HARD = 365  # fenêtre en jours pour la moyenne "12 derniers mois sur dur"

# Seuil minimum de matchs valides dans la fenêtre pour publier une moyenne.
# En-dessous, on considère l'estimation trop bruitée/peu fiable et on laisse
# le champ vide
MIN_HISTORY_FOR_AVERAGE = 3

METRICS = [
    "srv_1st_pct", "srv_2nd_pct", "srv_bp_saved", "ret_pts_pct", "ret_bp_conv",
    "hold_pct", "break_pct", "dominance_ratio", "pressure_rating"
]

# Demi-vie (en jours) de la pondération EMA — un match d'il y a HALF_LIFE_DAYS
# pèse deux fois moins qu'un match d'aujourd'hui. Fenêtre complémentaire aux
# moyennes r10
EMA_HALF_LIFE_DAYS = 45.0

# --- Étape 4bis : momentum / qualité d'adversaires --------------------------

# Demi-vie de l'EMA "longue" utilisée comme référence de niveau de fond pour
# détecter une tendance (progression/déclin) via delta_form_*. 120j ≈ un peu
# moins d'une demi-saison tennis : assez long pour lisser les creux/pics
# ponctuels, assez court pour ne pas mélanger deux saisons différentes.
EMA_LONG_HALF_LIFE_DAYS = 120.0

# Métriques sur lesquelles on calcule un momentum "intra-fenêtre r10"
# (pente 2e moitié vs 1re moitié). Limité à dominance_ratio et hold_pct :
# ce sont les deux résumés les plus synthétiques de la forme d'un joueur
# (respectivement "domination globale des points" et "solidité au service"),
# pas besoin de dupliquer le calcul sur les 9 métriques.
MOMENTUM_METRICS = ["dominance_ratio", "hold_pct"]

# Le momentum coupe la fenêtre r10 en deux moitiés : il faut donc au moins
# 2x le seuil normal pour avoir MIN_HISTORY_FOR_AVERAGE points valides de
# chaque côté de la coupure.
MIN_HISTORY_FOR_MOMENTUM = MIN_HISTORY_FOR_AVERAGE * 2

# Colonnes d'Elo pré-match à utiliser pour la qualité d'adversaire, par ordre
# de préférence. On privilégie l'Elo spécifique à la surface du match (plus
# pertinent pour juger "contre qui il a gagné sur CETTE surface"), avec
# repli sur l'Elo global si absent (matchs anciens / colonnes manquantes).
ELO_COLUMN_CANDIDATES = ["surface_elo_pre", "elo_pre"]


def to_float(value: str) -> float | None:
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def get_elo_pre(row: dict, prefix: str) -> float | None:
    """Renvoie l'Elo pré-match d'un joueur (prefix='winner' ou 'loser'),
    en essayant d'abord l'Elo par surface, puis l'Elo global. Renvoie None
    si aucune des deux colonnes n'est présente/valide (ex : build_elo.py
    pas encore lancé, ou colonnes nommées différemment) — dans ce cas les
    features de qualité d'adversaire ne seront simplement pas calculées."""
    for suffix in ELO_COLUMN_CANDIDATES:
        val = to_float(row.get(f"{prefix}_{suffix}", ""))
        if val is not None:
            return val
    return None


def compute_side_stats(row: dict, prefix: str) -> dict[str, float] | None:
    """Calcule les métriques de SERVICE pour un joueur (prefix='w' ou 'l')
    à partir de ses propres colonnes de stats. Renvoie None si les stats
    de base sont absentes (cas fréquent sur les matchs anciens/mineurs,
    cf. README section 2)."""
    svpt = to_float(row.get(f"{prefix}_svpt", ""))
    first_in = to_float(row.get(f"{prefix}_1stIn", ""))
    first_won = to_float(row.get(f"{prefix}_1stWon", ""))
    second_won = to_float(row.get(f"{prefix}_2ndWon", ""))
    bp_saved = to_float(row.get(f"{prefix}_bpSaved", ""))
    bp_faced = to_float(row.get(f"{prefix}_bpFaced", ""))
    sv_gms = to_float(row.get(f"{prefix}_SvGms", ""))

    if svpt is None or first_in is None or first_won is None or second_won is None:
        return None

    second_serve_pts = svpt - first_in
    srv_pts_won = first_won + second_won
    return {
        "svpt": svpt,
        "srv_1st_pct": safe_ratio(first_won, first_in),
        "srv_2nd_pct": safe_ratio(second_won, second_serve_pts),
        "srv_bp_saved": safe_ratio(bp_saved, bp_faced),
        "srv_pts_won": srv_pts_won,  # utile pour dériver le retour de l'adversaire
        "srv_pts_won_pct": safe_ratio(srv_pts_won, svpt),
        "bp_faced": bp_faced,
        "bp_saved": bp_saved,
        "sv_gms": sv_gms,
    }


def compute_match_player_stats(row: dict) -> dict[str, dict[str, float] | None]:
    """
    Pour un match donné, calcule les métriques de service/retour pour le
    gagnant et le perdant, plus 4 métriques dérivées :

      - hold_pct         : % de jeux de service conservés, approximé par
                            1 - (balles de break converties par l'adversaire
                            / nombre de jeux de service). Approximation
                            standard en l'absence de log jeu par jeu : elle
                            suppose qu'une balle de break convertie = un jeu
                            perdu, ce qui sous-estime légèrement le nombre
                            réel de breaks si plusieurs balles de break sont
                            jouées dans le même jeu avant de le perdre.
      - break_pct        : symétrique de hold_pct, côté retour (% des jeux
                            de service ADVERSES effectivement cassés).
      - dominance_ratio   : (% points gagnés au retour) / (% points PERDUS
                            au service) — > 1.0 indique un joueur qui domine
                            l'échange de points sur l'ensemble du match
                            (métrique popularisée par Craig O'Shannessy /
                            Tennis Abstract).
      - pressure_rating   : srv_bp_saved + ret_bp_conv — somme simple des
                            deux indicateurs de performance sur balle de
                            break, comme proxy de solidité mentale. NB : ne
                            capture pas spécifiquement les tie-breaks (donnée
                            indisponible au niveau match ; le Match Charting
                            Project, mentionné au README section 2, serait
                            la seule source point-par-point pour aller plus
                            loin sur ce point précis).

    Le retour d'un joueur se déduit des stats de SERVICE de l'adversaire.
    """
    w_serve = compute_side_stats(row, "w")
    l_serve = compute_side_stats(row, "l")

    result: dict[str, dict[str, float] | None] = {"winner": None, "loser": None}

    if w_serve is not None and l_serve is not None:
        # Retour du gagnant = performance au service du perdant, inversée.
        w_ret_pts_pct = safe_ratio(l_serve["svpt"] - l_serve["srv_pts_won"], l_serve["svpt"])
        w_games_broken_by_w = safe_ratio(
            (l_serve["bp_faced"] - l_serve["bp_saved"]) if (l_serve["bp_faced"] is not None and l_serve["bp_saved"] is not None) else None,
            l_serve["bp_faced"],
        )
        w_ret_bp_conv = w_games_broken_by_w
        w_hold_pct = None
        if w_serve["sv_gms"] not in (None, 0) and w_serve["bp_faced"] is not None and w_serve["bp_saved"] is not None:
            broken_ratio = safe_ratio(w_serve["bp_faced"] - w_serve["bp_saved"], w_serve["sv_gms"])
            w_hold_pct = 1.0 - broken_ratio if broken_ratio is not None else None
        w_break_pct = None
        if l_serve["sv_gms"] not in (None, 0) and l_serve["bp_faced"] is not None and l_serve["bp_saved"] is not None:
            w_break_pct = safe_ratio(l_serve["bp_faced"] - l_serve["bp_saved"], l_serve["sv_gms"])
        w_dominance = None
        if w_serve["srv_pts_won_pct"] is not None and w_ret_pts_pct is not None:
            srv_lost_pct = 1.0 - w_serve["srv_pts_won_pct"]
            w_dominance = safe_ratio(w_ret_pts_pct, srv_lost_pct)
        w_pressure = None
        if w_serve["srv_bp_saved"] is not None and w_ret_bp_conv is not None:
            w_pressure = w_serve["srv_bp_saved"] + w_ret_bp_conv

        result["winner"] = {
            "srv_1st_pct": w_serve["srv_1st_pct"],
            "srv_2nd_pct": w_serve["srv_2nd_pct"],
            "srv_bp_saved": w_serve["srv_bp_saved"],
            "ret_pts_pct": w_ret_pts_pct,
            "ret_bp_conv": w_ret_bp_conv,
            "hold_pct": w_hold_pct,
            "break_pct": w_break_pct,
            "dominance_ratio": w_dominance,
            "pressure_rating": w_pressure,
        }

        l_ret_pts_pct = safe_ratio(w_serve["svpt"] - w_serve["srv_pts_won"], w_serve["svpt"])
        l_ret_bp_conv = safe_ratio(
            (w_serve["bp_faced"] - w_serve["bp_saved"]) if (w_serve["bp_faced"] is not None and w_serve["bp_saved"] is not None) else None,
            w_serve["bp_faced"],
        )
        l_hold_pct = None
        if l_serve["sv_gms"] not in (None, 0) and l_serve["bp_faced"] is not None and l_serve["bp_saved"] is not None:
            broken_ratio_l = safe_ratio(l_serve["bp_faced"] - l_serve["bp_saved"], l_serve["sv_gms"])
            l_hold_pct = 1.0 - broken_ratio_l if broken_ratio_l is not None else None
        l_break_pct = None
        if w_serve["sv_gms"] not in (None, 0) and w_serve["bp_faced"] is not None and w_serve["bp_saved"] is not None:
            l_break_pct = safe_ratio(w_serve["bp_faced"] - w_serve["bp_saved"], w_serve["sv_gms"])
        l_dominance = None
        if l_serve["srv_pts_won_pct"] is not None and l_ret_pts_pct is not None:
            srv_lost_pct = 1.0 - l_serve["srv_pts_won_pct"]
            l_dominance = safe_ratio(l_ret_pts_pct, srv_lost_pct)
        l_pressure = None
        if l_serve["srv_bp_saved"] is not None and l_ret_bp_conv is not None:
            l_pressure = l_serve["srv_bp_saved"] + l_ret_bp_conv

        result["loser"] = {
            "srv_1st_pct": l_serve["srv_1st_pct"],
            "srv_2nd_pct": l_serve["srv_2nd_pct"],
            "srv_bp_saved": l_serve["srv_bp_saved"],
            "ret_pts_pct": l_ret_pts_pct,
            "ret_bp_conv": l_ret_bp_conv,
            "hold_pct": l_hold_pct,
            "break_pct": l_break_pct,
            "dominance_ratio": l_dominance,
            "pressure_rating": l_pressure,
        }

    return result


class PlayerHistory:
    """Historique glissant d'un joueur pour trois modes de lissage de forme
    (r10, hard12mo, ema courte), plus les extensions momentum/qualité :

    - `recent10`  : deque des 10 derniers matchs valides (toutes surfaces),
                    enrichis des champs 'result'/'own_elo_pre'/'opp_elo_pre'
                    utilisés pour le momentum et la qualité d'adversaire.
    - `ema`       : moyenne mobile exponentielle COURTE (demi-vie
                    EMA_HALF_LIFE_DAYS), pondérée par la récence en JOURS.
    - `ema_long`  : même principe, demi-vie EMA_LONG_HALF_LIFE_DAYS — sert
                    de référence de niveau de fond pour delta_form_*.
    - `streak`    : série de victoires/défaites consécutives (signée).
    - `quality_ema` : EMA de (elo_adversaire - elo_propre), calculée
                    uniquement sur les matchs GAGNÉS.
    """

    def __init__(self):
        self.recent10: deque[dict] = deque(maxlen=ROLLING_WINDOW_N)
        self.hard12mo: deque[tuple[str, dict]] = deque()  # (date_str, stats)

        self.ema_values: dict[str, float] = {}
        self.ema_last_date: datetime | None = None
        self.ema_metric_counts: dict[str, int] = defaultdict(int)

        # --- Étape 4bis ---
        self.ema_long_values: dict[str, float] = {}
        self.ema_long_last_date: datetime | None = None
        self.ema_long_metric_counts: dict[str, int] = defaultdict(int)

        self.streak: int = 0  # >0 = série de victoires, <0 = série de défaites

        self.quality_ema_value: float | None = None
        self.quality_ema_last_date: datetime | None = None
        self.quality_ema_n: int = 0  # nombre de VICTOIRES vues (dénominateur du seuil)

    def get_rolling_averages(self, current_date: str) -> dict[str, float | str | int]:
        """Renvoie les moyennes/indicateurs glissants PRE-match (avant tout
        ajout du match courant), pour tous les modes, avec les compteurs `_n`."""

        out: dict[str, float | str | int] = {}

        # --- r10 (inchangé) ---
        n_recent = len(self.recent10)
        out["r10_n"] = n_recent
        for metric in METRICS:
            values = [m[metric] for m in self.recent10 if m.get(metric) is not None]
            if len(values) >= MIN_HISTORY_FOR_AVERAGE:
                out[f"r10_{metric}"] = round(sum(values) / len(values), 4)
            else:
                out[f"r10_{metric}"] = ""

        # --- EMA courte (inchangé) ---
        for metric in METRICS:
            if self.ema_metric_counts[metric] >= MIN_HISTORY_FOR_AVERAGE and metric in self.ema_values:
                out[f"ema_{metric}"] = round(self.ema_values[metric], 4)
            else:
                out[f"ema_{metric}"] = ""
        out["ema_n"] = self.ema_metric_counts["srv_1st_pct"]  # référence pour lisibilité du rapport

        # --- Étape 4bis : EMA longue + delta_form ---
        for metric in METRICS:
            if self.ema_long_metric_counts[metric] >= MIN_HISTORY_FOR_AVERAGE and metric in self.ema_long_values:
                out[f"ema_long_{metric}"] = round(self.ema_long_values[metric], 4)
            else:
                out[f"ema_long_{metric}"] = ""
        out["ema_long_n"] = self.ema_long_metric_counts["srv_1st_pct"]

        for metric in METRICS:
            short_val = out.get(f"ema_{metric}")
            long_val = out.get(f"ema_long_{metric}")
            if short_val not in ("", None) and long_val not in ("", None):
                out[f"delta_form_{metric}"] = round(short_val - long_val, 4)
            else:
                out[f"delta_form_{metric}"] = ""

        # --- Étape 4bis : momentum intra-fenêtre r10 (pente 2e moitié - 1re moitié) ---
        for metric in MOMENTUM_METRICS:
            values = [m[metric] for m in self.recent10 if m.get(metric) is not None]
            if len(values) >= MIN_HISTORY_FOR_MOMENTUM:
                mid = len(values) // 2
                first_half = values[:mid]
                second_half = values[mid:]
                if len(first_half) >= MIN_HISTORY_FOR_AVERAGE and len(second_half) >= MIN_HISTORY_FOR_AVERAGE:
                    out[f"momentum_{metric}"] = round(
                        (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half)), 4
                    )
                else:
                    out[f"momentum_{metric}"] = ""
            else:
                out[f"momentum_{metric}"] = ""

        # --- Étape 4bis : streak (valeur brute, pas de seuil d'historique) ---
        out["streak"] = self.streak

        # --- Étape 4bis : qualité des adversaires sur la fenêtre r10 ---
        wins_in_window = [m for m in self.recent10 if m.get("result") == 1 and m.get("opp_elo_pre") is not None]
        losses_in_window = [m for m in self.recent10 if m.get("result") == 0
                             and m.get("opp_elo_pre") is not None and m.get("own_elo_pre") is not None]
        wins_with_own_elo = [m for m in wins_in_window if m.get("own_elo_pre") is not None]

        if len(wins_in_window) >= MIN_HISTORY_FOR_AVERAGE:
            out["r10_avg_opp_elo_beaten"] = round(
                sum(m["opp_elo_pre"] for m in wins_in_window) / len(wins_in_window), 2
            )
        else:
            out["r10_avg_opp_elo_beaten"] = ""

        if len(wins_with_own_elo) >= MIN_HISTORY_FOR_AVERAGE:
            n_upsets = sum(1 for m in wins_with_own_elo if m["opp_elo_pre"] > m["own_elo_pre"])
            out["r10_upset_win_pct"] = round(n_upsets / len(wins_with_own_elo), 4)
        else:
            out["r10_upset_win_pct"] = ""

        if len(losses_in_window) >= MIN_HISTORY_FOR_AVERAGE:
            n_bad_losses = sum(1 for m in losses_in_window if m["opp_elo_pre"] < m["own_elo_pre"])
            out["r10_bad_loss_pct"] = round(n_bad_losses / len(losses_in_window), 4)
        else:
            out["r10_bad_loss_pct"] = ""

        # --- Étape 4bis : quality_ema (EMA de la marge d'Elo sur les victoires) ---
        if self.quality_ema_n >= MIN_HISTORY_FOR_AVERAGE and self.quality_ema_value is not None:
            out["quality_ema"] = round(self.quality_ema_value, 2)
        else:
            out["quality_ema"] = ""
        out["quality_ema_n"] = self.quality_ema_n

        return out

    def _clean_hard12mo(self, current_date: datetime) -> None:
        """Conserve uniquement les matchs joués sur dur dans les 365 derniers jours."""
        while self.hard12mo:
            match_date, _ = self.hard12mo[0]
            try:
                dt = datetime.strptime(match_date, "%Y-%m-%d")
                if (current_date - dt).days > ROLLING_WINDOW_DAYS_HARD:
                    self.hard12mo.popleft()
                else:
                    break
            except ValueError:
                self.hard12mo.popleft()

    @staticmethod
    def _ema_alpha(days_elapsed: int, half_life_days: float) -> float:
        """Poids donné au NOUVEAU point pour une EMA pondérée par le temps.
        days_elapsed=0 (deux matchs le même jour, ex: double relancé) retombe
        sur un alpha fixe de 0.5 plutôt qu'un alpha nul qui ignorerait
        totalement le nouveau match."""
        if days_elapsed <= 0:
            return 0.5
        return 1.0 - 0.5 ** (days_elapsed / half_life_days)

    def add_match(self, date_str: str, surface_norm: str, stats: dict) -> None:
        """`stats` contient les métriques de service/retour habituelles,
        PLUS (Étape 4bis) 'result' (1=victoire/0=défaite), 'own_elo_pre' et
        'opp_elo_pre' (peuvent être None si l'Elo n'était pas disponible)."""
        self.recent10.append(stats)

        current = None
        try:
            current = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass

        if current is not None:
            # --- EMA courte (inchangé) ---
            if self.ema_last_date is None:
                for metric in METRICS:
                    if stats.get(metric) is not None:
                        self.ema_values[metric] = stats[metric]
                        self.ema_metric_counts[metric] += 1
            else:
                days_elapsed = max(0, (current - self.ema_last_date).days)
                alpha = self._ema_alpha(days_elapsed, EMA_HALF_LIFE_DAYS)
                for metric in METRICS:
                    if stats.get(metric) is not None:
                        old = self.ema_values.get(metric, stats[metric])
                        self.ema_values[metric] = alpha * stats[metric] + (1 - alpha) * old
                        self.ema_metric_counts[metric] += 1
                    # Si la métrique est absente pour CE match (ex: SvGms
                    # manquant), on ne touche ni à sa valeur EMA ni à son
                    # compteur — elle reste "en attente" de la prochaine
                    # donnée valide plutôt que d'être polluée par une
                    # décroissance temporelle sans nouvelle observation.
            self.ema_last_date = current

            # --- Étape 4bis : EMA longue (même logique, demi-vie différente) ---
            if self.ema_long_last_date is None:
                for metric in METRICS:
                    if stats.get(metric) is not None:
                        self.ema_long_values[metric] = stats[metric]
                        self.ema_long_metric_counts[metric] += 1
            else:
                days_elapsed_long = max(0, (current - self.ema_long_last_date).days)
                alpha_long = self._ema_alpha(days_elapsed_long, EMA_LONG_HALF_LIFE_DAYS)
                for metric in METRICS:
                    if stats.get(metric) is not None:
                        old_long = self.ema_long_values.get(metric, stats[metric])
                        self.ema_long_values[metric] = alpha_long * stats[metric] + (1 - alpha_long) * old_long
                        self.ema_long_metric_counts[metric] += 1
            self.ema_long_last_date = current

            # --- Étape 4bis : quality_ema, uniquement mis à jour sur les victoires ---
            result = stats.get("result")
            own_elo = stats.get("own_elo_pre")
            opp_elo = stats.get("opp_elo_pre")
            if result == 1 and own_elo is not None and opp_elo is not None:
                margin = opp_elo - own_elo  # >0 = a battu un adversaire mieux coté
                if self.quality_ema_last_date is None or self.quality_ema_value is None:
                    self.quality_ema_value = margin
                else:
                    days_elapsed_q = max(0, (current - self.quality_ema_last_date).days)
                    alpha_q = self._ema_alpha(days_elapsed_q, EMA_HALF_LIFE_DAYS)
                    self.quality_ema_value = alpha_q * margin + (1 - alpha_q) * self.quality_ema_value
                self.quality_ema_last_date = current
                self.quality_ema_n += 1

            # --- Étape 4bis : streak ---
            if result == 1:
                self.streak = self.streak + 1 if self.streak >= 0 else 1
            elif result == 0:
                self.streak = self.streak - 1 if self.streak <= 0 else -1


def process(matches: list[dict]) -> tuple[list[dict], dict]:
    histories: dict[str, PlayerHistory] = defaultdict(PlayerHistory)
    enriched_rows: list[dict] = []
    stats_counter = {
        "n_matches": 0,
        "n_missing_serve_stats": 0,
        "n_missing_sv_gms": 0,
        "n_missing_elo_for_quality": 0,  # Étape 4bis : au moins un Elo pré-match absent
    }

    for row in matches:
        winner_id = (row.get("winner_id") or "").strip()
        loser_id = (row.get("loser_id") or "").strip()
        date_str = (row.get("tourney_date") or "").strip()
        surface_norm = (row.get("surface_norm") or "").strip()

        new_row = dict(row)

        if winner_id and date_str:
            w_hist = histories[winner_id]
            w_pre = w_hist.get_rolling_averages(date_str)
            for k, v in w_pre.items():
                new_row[f"w_{k}"] = v

        if loser_id and date_str:
            l_hist = histories[loser_id]
            l_pre = l_hist.get_rolling_averages(date_str)
            for k, v in l_pre.items():
                new_row[f"l_{k}"] = v

        if to_float(row.get("w_SvGms", "")) is None or to_float(row.get("l_SvGms", "")) is None:
            stats_counter["n_missing_sv_gms"] += 1

        # --- Étape 4bis : Elo pré-match des deux joueurs (pour qualité d'adversaire) ---
        winner_elo_pre = get_elo_pre(row, "winner")
        loser_elo_pre = get_elo_pre(row, "loser")
        if winner_elo_pre is None or loser_elo_pre is None:
            stats_counter["n_missing_elo_for_quality"] += 1

        match_stats = compute_match_player_stats(row)
        if match_stats["winner"] is None or match_stats["loser"] is None:
            stats_counter["n_missing_serve_stats"] += 1
        else:
            if winner_id and date_str:
                winner_stats = dict(match_stats["winner"])
                winner_stats["result"] = 1
                winner_stats["own_elo_pre"] = winner_elo_pre
                winner_stats["opp_elo_pre"] = loser_elo_pre
                histories[winner_id].add_match(date_str, surface_norm, winner_stats)
            if loser_id and date_str:
                loser_stats = dict(match_stats["loser"])
                loser_stats["result"] = 0
                loser_stats["own_elo_pre"] = loser_elo_pre
                loser_stats["opp_elo_pre"] = winner_elo_pre
                histories[loser_id].add_match(date_str, surface_norm, loser_stats)

        stats_counter["n_matches"] += 1
        enriched_rows.append(new_row)

    return enriched_rows, stats_counter


def write_output(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    # Colonnes brutes de service/retour à exclure du CSV final
    RAW_STATS_TO_EXCLUDE = {
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", 
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", 
        "l_SvGms", "l_bpSaved", "l_bpFaced"
    }

    # Conservation uniquement des colonnes hors liste d'exclusion
    fieldnames = [col for col in rows[0].keys() if col not in RAW_STATS_TO_EXCLUDE]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict], stats_counter: dict, output_path: Path) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    n_total = len(rows)

    def coverage(mode: str, metric: str) -> int:
        return sum(1 for r in rows if r.get(f"w_{mode}_{metric}") not in ("", None))

    def coverage_field(field: str) -> int:
        return sum(1 for r in rows if r.get(f"w_{field}") not in ("", None))

    modes = [("r10", "10 derniers matchs, toutes surfaces"),
              ("hard12mo", "12 derniers mois, sur dur"),
              ("ema", f"EMA courte, demi-vie {EMA_HALF_LIFE_DAYS:.0f}j"),
              ("ema_long", f"EMA longue, demi-vie {EMA_LONG_HALF_LIFE_DAYS:.0f}j"),
              ("delta_form", "delta EMA courte - EMA longue")]

    lines = [
        "# Rapport — build_features.py",
        f"\nGénéré le {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC\n",
        f"**Matchs traités :** {stats_counter['n_matches']}",
        f"**Matchs sans stats de service exploitables (ignorés pour l'historique, "
        f"conservés dans la sortie) :** {stats_counter['n_missing_serve_stats']}",
        f"**Matchs avec `SvGms` manquant sur au moins un camp** (affecte `hold_pct`/`break_pct` "
        f"uniquement, pas les autres métriques) : {stats_counter['n_missing_sv_gms']} "
        f"({round(100*stats_counter['n_missing_sv_gms']/max(1,n_total),2)}%)",
        f"**Matchs avec Elo pré-match manquant sur au moins un camp** (bloque les features "
        f"de qualité d'adversaire pour ce match) : {stats_counter['n_missing_elo_for_quality']} "
        f"({round(100*stats_counter['n_missing_elo_for_quality']/max(1,n_total),2)}%)",
        f"**Seuil minimum de matchs pour publier une moyenne :** {MIN_HISTORY_FOR_AVERAGE} "
        f"(x2 = {MIN_HISTORY_FOR_MOMENTUM} pour le momentum, qui coupe la fenêtre en 2)\n",
        "## Couverture par mode de lissage (nombre de lignes avec valeur disponible, côté gagnant)",
        "| Mode | " + " | ".join(METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
    ]
    for mode, _ in modes:
        row_vals = [str(coverage(mode, m)) for m in METRICS]
        lines.append(f"| {mode} | " + " | ".join(row_vals) + f" | *(/{n_total})*")

    lines += [
        "\n## Couverture — momentum, streak, qualité d'adversaire (côté gagnant)",
        "| Feature | Lignes disponibles |",
        "|---|---|",
    ]
    for metric in MOMENTUM_METRICS:
        lines.append(f"| momentum_{metric} | {coverage_field(f'momentum_{metric}')} / {n_total} |")
    lines += [
        f"| streak | {n_total} / {n_total} *(toujours renseigné, 0 par défaut sans historique)* |",
        f"| r10_avg_opp_elo_beaten | {coverage_field('r10_avg_opp_elo_beaten')} / {n_total} |",
        f"| r10_upset_win_pct | {coverage_field('r10_upset_win_pct')} / {n_total} |",
        f"| r10_bad_loss_pct | {coverage_field('r10_bad_loss_pct')} / {n_total} |",
        f"| quality_ema | {coverage_field('quality_ema')} / {n_total} |",
    ]

    lines += [
        "\n## Détail des modes de lissage",
    ]
    for mode, desc in modes:
        n_avail = coverage(mode, "srv_1st_pct") if mode != "delta_form" else coverage_field("delta_form_srv_1st_pct")
        lines.append(f"- **{mode}** ({desc}) : {n_avail} / {n_total} lignes disponibles (côté gagnant, "
                      f"référence sur `srv_1st_pct` — les autres métriques peuvent différer légèrement, "
                      f"voir tableau ci-dessus, notamment `hold_pct`/`break_pct` à cause de `SvGms`).")

    lines += [
        "\n## Colonnes ajoutées (préfixe `w_`/`l_`, pour chacun des modes `r10_`/`hard12mo_`/`ema_`/`ema_long_`/`delta_form_`)",
        "- `*_n` : nombre de matchs pris en compte dans la fenêtre/l'EMA",
        "- `*_srv_1st_pct`, `*_srv_2nd_pct`, `*_srv_bp_saved` : performance au service",
        "- `*_ret_pts_pct`, `*_ret_bp_conv` : performance au retour",
        "- `*_hold_pct` : % de jeux de service conservés (approximation via balles de break, voir code)",
        "- `*_break_pct` : % des jeux de service adverses cassés",
        "- `*_dominance_ratio` : % retour gagné / % service perdu — > 1.0 = domine l'échange de points",
        "- `*_pressure_rating` : `srv_bp_saved + ret_bp_conv`, proxy de solidité sur balle de break "
        "(ne capture pas spécifiquement les tie-breaks, donnée indisponible à ce niveau)",
        "\n## Colonnes ajoutées — momentum / qualité d'adversaire (Étape 4bis, préfixe `w_`/`l_`)",
        "- `momentum_dominance_ratio`, `momentum_hold_pct` : pente intra-fenêtre r10 "
        "(moyenne 2e moitié - moyenne 1re moitié). Positif = en progression sur cette métrique.",
        "- `delta_form_*` : EMA courte - EMA longue, pour chacune des 9 métriques. Positif = "
        "le joueur performe actuellement au-dessus de son niveau de fond récent.",
        "- `streak` : série de victoires (positif) ou défaites (négatif) consécutives, avant le match.",
        "- `r10_avg_opp_elo_beaten` : Elo moyen (surface si dispo, sinon global) des adversaires "
        "battus sur les 10 derniers matchs.",
        "- `r10_upset_win_pct` : % de ces victoires obtenues contre un adversaire mieux coté.",
        "- `r10_bad_loss_pct` : % des défaites de la fenêtre concédées contre un adversaire moins bien coté.",
        "- `quality_ema` / `quality_ema_n` : EMA (et compteur) de la marge d'Elo "
        "(adversaire - soi-même) sur les seuls matchs GAGNÉS, pondérée par la récence.",
        "\nUne cellule vide signifie : historique insuffisant "
        f"(< {MIN_HISTORY_FOR_AVERAGE} matchs valides, ou < {MIN_HISTORY_FOR_MOMENTUM} pour le momentum), "
        "pas une valeur de zéro. `streak` fait exception : il vaut 0 par défaut (aucun historique).",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ Rapport écrit : {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcule les features de service/retour lissées.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_FILE),
                         help="Fichier d'entrée (sortie de build_elo.py). "
                              f"Défaut : {DEFAULT_INPUT_FILE.name}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROCESSED_DIR / input_path.name if "/" not in args.input else Path(args.input)

    print("ÉTAPE 4 — Features de service/retour lissées + momentum/qualité d'adversaire")

    if not input_path.exists():
        print(f"✗ {input_path} introuvable. Lance d'abord scripts/build_elo.py "
              f"(avec --k-mode dynamic --suffix _dynamic pour matcher le défaut de ce script).")
        return 1

    with input_path.open("r", encoding="utf-8", newline="") as f:
        matches = list(csv.DictReader(f))
    print(f"→ {len(matches)} matchs chargés depuis {input_path.name}.")

    enriched_rows, stats_counter = process(matches)
    print(f"→ {stats_counter['n_matches']} matchs traités, "
          f"{stats_counter['n_missing_serve_stats']} sans stats de service exploitables, "
          f"{stats_counter['n_missing_elo_for_quality']} sans Elo complet pour la qualité d'adversaire.")

    write_output(enriched_rows, OUTPUT_FILE)
    print(f"→ Fichier enrichi écrit : {OUTPUT_FILE}")

    write_report(enriched_rows, stats_counter, OUTPUT_REPORT_FILE)

    return 0


if __name__ == "__main__":
    sys.exit(main())