"""
build_fatigue.py
=================
Étape 5 du projet : fatigue et forme récente.

Pour chaque match, calcule PAR JOUEUR des indicateurs de charge GÉNÉRIQUES,
valables pour n'importe quel tournoi :

  - `rest_days`           : nombre de jours depuis le match précédent du
                            joueur (vide si c'est son premier match connu).
  - `matches_last_30d`, `sets_last_30d`, `minutes_last_30d` (+ `_n`) :
                            charge sur les 30 derniers jours.
  - `fatigue_ema_sets`, `fatigue_ema_minutes` : charge accumulée par Exponential Moving
                            Average avec décroissance temporelle (half-life).

Le score est parsé pour compter les sets joués (ex: "6-4 7-6(5) 3-6" -> 3
sets), ce qui fonctionne même quand `minutes` est manquant.

Usage:
    python scripts/build_fatigue.py
    python scripts/build_fatigue.py --input atp_matches_with_features.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "fatigue"

DEFAULT_INPUT_FILE = PROCESSED_DIR / "atp_matches_with_features.csv"
OUTPUT_FILE = PROCESSED_DIR / "atp_matches_with_fatigue.csv"
OUTPUT_REPORT_FILE = OUTPUTS_DIR / "build_fatigue_report.md"

# Fenêtre glissante
WINDOW_DAYS_SHORT = 30

# Paramètres EMA (décroissance temporelle)
HALF_LIFE_DAYS = 14.0  # Demi-vie de la fatigue en jours
DECAY_RATE = math.log(2) / HALF_LIFE_DAYS

SET_SCORE_PATTERN = re.compile(r"^\d+-\d+(\(\d+\))?$")


def count_sets(score: str) -> int | None:
    """Compte le nombre de sets joués à partir de la chaîne `score`.
    Renvoie None si le score est vide/illisible (ex: 'W/O') plutôt que 0,
    pour ne pas sous-compter silencieusement une charge de match réelle."""
    score = (score or "").strip()
    if not score:
        return None
    tokens = score.replace("RET", "").replace("DEF", "").split()
    n_sets = sum(1 for t in tokens if SET_SCORE_PATTERN.match(t.strip()))
    return n_sets if n_sets > 0 else None


def to_date(date_str: str) -> datetime | None:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


class PlayerFatigueHistory:
    """Historique glissant et suivi de l'EMA de fatigue d'un joueur."""

    def __init__(self):
        # Chaque entrée : (date, sets_played_or_None, minutes_or_None)
        self.matches: deque[tuple[datetime, int | None, float | None]] = deque()
        self.last_match_date: datetime | None = None
        
        # Accumulateurs EMA
        self.ema_sets: float = 0.0
        self.ema_minutes: float = 0.0

    def _entries_within(self, current: datetime, window_days: int) -> list[tuple[datetime, int | None, float | None]]:
        return [e for e in self.matches if (current - e[0]).days <= window_days]

    @staticmethod
    def _summarize(entries: list[tuple[datetime, int | None, float | None]]) -> dict:
        n_matches = len(entries)
        sets_sum = sum(s for _, s, _ in entries if s is not None)
        minutes_values = [m for _, _, m in entries if m is not None]
        minutes_sum = round(sum(minutes_values), 1) if minutes_values else ""
        return {
            "matches": n_matches,
            "sets": sets_sum,
            "minutes": minutes_sum,
            "minutes_n": len(minutes_values),
        }

    def get_pre_match_features(self, current_date_str: str) -> dict:
        current = to_date(current_date_str)
        out: dict = {
            "rest_days": "",
            "matches_last_30d": 0,
            "sets_last_30d": 0,
            "minutes_last_30d": "",
            "minutes_last_30d_n": 0,
            "fatigue_ema_sets": 0.0,
            "fatigue_ema_minutes": 0.0,
        }
        if current is None:
            return out

        if self.last_match_date is not None:
            delta_days = (current - self.last_match_date).days
            out["rest_days"] = delta_days

            # Application de la décroissance temporelle à l'état EMA actuel
            decay_factor = math.exp(-DECAY_RATE * max(0, delta_days))
            current_ema_sets = self.ema_sets * decay_factor
            current_ema_minutes = self.ema_minutes * decay_factor
        else:
            current_ema_sets = 0.0
            current_ema_minutes = 0.0

        out["fatigue_ema_sets"] = round(current_ema_sets, 3)
        out["fatigue_ema_minutes"] = round(current_ema_minutes, 2)

        short = self._summarize(self._entries_within(current, WINDOW_DAYS_SHORT))
        out["matches_last_30d"] = short["matches"]
        out["sets_last_30d"] = short["sets"]
        out["minutes_last_30d"] = short["minutes"]
        out["minutes_last_30d_n"] = short["minutes_n"]

        return out

    def add_match(self, date_str: str, sets: int | None, minutes: float | None) -> None:
        current = to_date(date_str)
        if current is None:
            return

        # Mise à jour EMA avec décroissance temporelle
        if self.last_match_date is not None:
            delta_days = max(0, (current - self.last_match_date).days)
            decay_factor = math.exp(-DECAY_RATE * delta_days)
            self.ema_sets *= decay_factor
            self.ema_minutes *= decay_factor

        # Ajout de la charge du match actuel
        if sets is not None:
            self.ema_sets += sets
        if minutes is not None:
            self.ema_minutes += minutes

        self.matches.append((current, sets, minutes))
        self.last_match_date = current

        # Purge de l'historique brut
        cutoff = current
        while self.matches and (cutoff - self.matches[0][0]).days > 70:
            self.matches.popleft()


def to_minutes(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def process(matches: list[dict]) -> tuple[list[dict], dict]:
    histories: dict[str, PlayerFatigueHistory] = defaultdict(PlayerFatigueHistory)
    enriched_rows: list[dict] = []
    stats_counter = {"n_matches": 0, "n_missing_minutes": 0, "n_missing_score_for_sets": 0}

    for row in matches:
        winner_id = (row.get("winner_id") or "").strip()
        loser_id = (row.get("loser_id") or "").strip()
        date_str = (row.get("tourney_date") or "").strip()

        new_row = dict(row)

        if winner_id and date_str:
            w_feats = histories[winner_id].get_pre_match_features(date_str)
            for k, v in w_feats.items():
                new_row[f"w_{k}"] = v
        if loser_id and date_str:
            l_feats = histories[loser_id].get_pre_match_features(date_str)
            for k, v in l_feats.items():
                new_row[f"l_{k}"] = v

        minutes = to_minutes(row.get("minutes", ""))
        sets = count_sets(row.get("score", ""))

        if minutes is None:
            stats_counter["n_missing_minutes"] += 1
        if sets is None:
            stats_counter["n_missing_score_for_sets"] += 1

        if winner_id and date_str:
            histories[winner_id].add_match(date_str, sets, minutes)
        if loser_id and date_str:
            histories[loser_id].add_match(date_str, sets, minutes)

        stats_counter["n_matches"] += 1
        enriched_rows.append(new_row)

    return enriched_rows, stats_counter


def write_output(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict], stats_counter: dict, output_path: Path) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    n_total = len(rows)
    pct_missing_minutes = round(100 * stats_counter["n_missing_minutes"] / n_total, 2) if n_total else 0
    pct_missing_sets = round(100 * stats_counter["n_missing_score_for_sets"] / n_total, 2) if n_total else 0

    n_with_rest_days = sum(1 for r in rows if r.get("w_rest_days") != "")

    lines = [
        "# Rapport — build_fatigue.py",
        f"\nGénéré le {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC\n",
        f"**Matchs traités :** {n_total}",
        f"**Matchs avec `minutes` manquant :** {stats_counter['n_missing_minutes']} ({pct_missing_minutes}%)",
        f"**Matchs avec score illisible pour compter les sets :** {stats_counter['n_missing_score_for_sets']} ({pct_missing_sets}%)",
        f"**Lignes avec `rest_days` disponible (côté gagnant, i.e. pas un 1er match connu) :** {n_with_rest_days} / {n_total}\n",
        "## Colonnes ajoutées (préfixe `w_`/`l_`)",
        "- `rest_days` : jours depuis le match précédent (vide si 1er match connu du joueur)",
        f"- `matches_last_30d`, `sets_last_30d`, `minutes_last_30d` (+ `minutes_last_30d_n`) : charge sur les {WINDOW_DAYS_SHORT} derniers jours",
        f"- `fatigue_ema_sets`, `fatigue_ema_minutes` : charge accumulée sous EMA (demi-vie = {HALF_LIFE_DAYS} jours)",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ Rapport écrit : {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcule les features de fatigue/forme récente.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_FILE),
                        help=f"Fichier d'entrée (sortie de build_features.py). Défaut : {DEFAULT_INPUT_FILE.name}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_absolute() and "/" not in args.input:
        input_path = PROCESSED_DIR / args.input

    print("ÉTAPE 5 — Fatigue et forme récente (avec EMA)")

    if not input_path.exists():
        print(f"✗ {input_path} introuvable. Lance d'abord scripts/build_features.py.")
        return 1

    with input_path.open("r", encoding="utf-8", newline="") as f:
        matches = list(csv.DictReader(f))
    print(f"→ {len(matches)} matchs chargés depuis {input_path.name}.")

    enriched_rows, stats_counter = process(matches)
    print(f"→ {stats_counter['n_matches']} matchs traités.")
    print(f"→ `minutes` manquant sur {stats_counter['n_missing_minutes']} matchs.")
    print(f"→ Score illisible pour compter les sets sur {stats_counter['n_missing_score_for_sets']} matchs.")

    write_output(enriched_rows, OUTPUT_FILE)
    print(f"→ Fichier enrichi écrit : {OUTPUT_FILE}")

    write_report(enriched_rows, stats_counter, OUTPUT_REPORT_FILE)

    return 0


if __name__ == "__main__":
    sys.exit(main())