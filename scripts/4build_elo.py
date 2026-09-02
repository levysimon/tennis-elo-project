"""
build_elo.py
============
Étape 3 du projet : calcul de l'Elo global, par surface, par
environnement (indoor/outdoor), et un "Clutch Elo" pour les matchs allés
au bout des 3 ou 5 (set décisif joué).

Principe :
Le fichier data/processed/atp_matches_all.csv est déjà trié par
(tourney_date, match_num). On parcourt les matchs UNE SEULE FOIS, dans cet
ordre, et pour chaque match :
  1. On lit les ratings Elo courants des deux joueurs (= état construit à
     partir de tous les matchs strictement antérieurs).
  2. On enregistre ces ratings "pre-match" dans la table de sortie — ce sont
     eux qui serviront de features pour la modélisation, jamais les ratings
     post-match.
  3. On met à jour les ratings après coup, en utilisant le résultat du match
     qu'on vient de lire.

Quatre familles de ratings sont maintenues en parallèle pour chaque joueur :
  - Elo global (toutes surfaces confondues)
  - Elo par surface (Hard / Clay / Grass)
  - Elo par environnement (surface + indoor/outdoor) — utile car la vitesse de balle et les conditions
    changent significativement le jeu, en particulier sur dur.
  - "Clutch Elo" : un Elo global séparé, mis à jour UNIQUEMENT sur les
    matchs allés au bout (nombre de sets joués == best_of, ex: 3 sets sur un
    match en 3, 5 sets sur un match en 5) - sert à capter la solidité dans
    les matchs serrés indépendamment du niveau général du joueur.

Un joueur non encore vu démarre à INITIAL_ELO (1500), la convention standard.

Usage:
    python scripts/build_elo.py
    python scripts/build_elo.py --k-mode fixed --k-factor 24
    python scripts/build_elo.py --no-split-indoor --no-clutch-elo
"""

from __future__ import annotations
import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "elo"

INPUT_FILE = PROCESSED_DIR / "atp_matches_with_odds.csv"
OUTPUT_MATCHES_FILE = PROCESSED_DIR / "atp_matches_with_elo.csv"
OUTPUT_RATINGS_FILE = OUTPUTS_DIR / "elo_ratings_final.csv"
OUTPUT_REPORT_FILE = OUTPUTS_DIR / "build_elo_report.md"

INITIAL_ELO = 1500.0
DEFAULT_K = 32.0

# Paramètres de la variante K dynamique, inspirée de l'approche FiveThirtyEight :
# K décroît avec le nombre de matchs déjà joués par le joueur, pour donner
# plus de poids aux premiers matchs (rating encore peu fiable, doit converger
# vite) et moins de poids une fois la carrière établie (rating déjà stable,
# on ne veut pas qu'un simple mauvais jour le fasse trop bouger).
#   K(m) = K_BASE / (m + K_OFFSET) ** K_EXPONENT
# Un plancher/plafond (K_MIN / K_MAX) évite les valeurs extrêmes en début de
# carrière (m=0) ou des valeurs trop faibles en fin de carrière très longue.
DYNAMIC_K_BASE = 250.0
DYNAMIC_K_OFFSET = 5.0
DYNAMIC_K_EXPONENT = 0.4
DYNAMIC_K_MIN = 16.0
DYNAMIC_K_MAX = 50.0

# Surfaces sur lesquelles on maintient un Elo dédié. "Unknown" est
# volontairement exclu : on ne veut pas mélanger un Elo par surface avec des
# matchs dont on ne sait pas sur quelle surface ils ont été joués.

TRACKED_SURFACES = ["Hard", "Clay", "Grass"]

SET_SCORE_PATTERN = re.compile(r"^\d+-\d+(\(\d+\))?$")


def normalize_indoor(raw_value: str) -> str:
    """Normalise la colonne `indoor` du dataset brut, dont le format exact
    peut varier (Yes/No, 1/0, Indoor/Outdoor...), vers 'Indoor'/'Outdoor'.
    Renvoie 'Unknown' si la valeur est absente ou non reconnue — dans ce
    cas, l'Elo par environnement n'est ni lu ni mis à jour pour ce match
    (même logique que pour une surface non suivie)."""
    value = (raw_value or "").strip().lower()
    if value in ("yes", "1", "true", "indoor", "i"):
        return "Indoor"
    if value in ("no", "0", "false", "outdoor", "o"):
        return "Outdoor"
    return "Unknown"


def count_sets(score: str) -> int | None:
    """Compte le nombre de sets joués à partir du score. Renvoie None si le
    score est illisible (walkover, retrait précoce sans set complet...),
    pour ne jamais assimiler un score illisible à '0 set' ou à un match
    décisif par défaut."""
    score = (score or "").strip()
    if not score:
        return None
    tokens = score.replace("RET", "").replace("DEF", "").split()
    n_sets = sum(1 for t in tokens if SET_SCORE_PATTERN.match(t.strip()))
    return n_sets if n_sets > 0 else None


def is_decisive_match(score: str, best_of: str) -> bool:
    """Un match est considéré 'allé au bout' si le nombre de sets joués
    est égal au maximum possible pour le format (best_of). Cette définition
    capture aussi bien un 7-6 7-6 en 3 sets qu'un 5e set en Grand Chelem —
    dans les deux cas, le joueur n'a pas pu s'économiser, la pression du
    dernier set a joué à plein."""
    n_sets = count_sets(score)
    try:
        n_best_of = int(best_of)
    except (TypeError, ValueError):
        return False
    return n_sets is not None and n_sets == n_best_of


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilité de victoire de A face à B selon la formule Elo standard."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


class EloTracker:
    """Maintient les ratings Elo (global + par surface + par environnement
    + clutch) de tous les joueurs, et applique la mise à jour standard
    après chaque match.

    k_mode="fixed"   : utilise k_factor pour tout le monde, tout le temps.
    k_mode="dynamic" : chaque joueur a son propre K, fonction du nombre de
                       matchs qu'il a déjà joués (approche FiveThirtyEight).
                       Le K est recalculé à CHAQUE match à partir du compteur
                       de matchs pré-match du joueur concerné (donc encore une
                       fois: aucune fuite d'information future)."""

    def __init__(self, k_factor: float = DEFAULT_K, k_mode: str = "dynamic",
                 split_indoor: bool = True, clutch_elo: bool = True):
        self.k_factor = k_factor
        self.k_mode = k_mode
        self.split_indoor = split_indoor
        self.clutch_elo_enabled = clutch_elo

        self.overall: dict[str, float] = defaultdict(lambda: INITIAL_ELO)
        self.by_surface: dict[str, dict[str, float]] = {
            surface: defaultdict(lambda: INITIAL_ELO) for surface in TRACKED_SURFACES
        }
        # Clé = "Hard-Indoor", "Hard-Outdoor", "Clay-Outdoor", etc. Créée à
        # la demande (defaultdict), pas de liste figée à l'avance : certaines
        # combinaisons (ex: Grass-Indoor) sont quasi inexistantes en réalité
        # et n'ont pas besoin d'exister tant qu'aucun match ne les déclenche.
        self.by_surface_env: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: INITIAL_ELO))
        self.clutch: dict[str, float] = defaultdict(lambda: INITIAL_ELO)

        self.matches_played: dict[str, int] = defaultdict(int)

    def get_overall(self, player_id: str) -> float:
        return self.overall[player_id]

    def get_surface(self, player_id: str, surface: str) -> float | None:
        if surface not in self.by_surface:
            return None
        return self.by_surface[surface][player_id]

    def get_surface_env(self, player_id: str, surface: str, indoor_label: str) -> float | None:
        if not self.split_indoor or surface not in TRACKED_SURFACES or indoor_label == "Unknown":
            return None
        key = f"{surface}-{indoor_label}"
        return self.by_surface_env[key][player_id]

    def get_clutch(self, player_id: str) -> float | None:
        if not self.clutch_elo_enabled:
            return None
        return self.clutch[player_id]

    def k_for(self, player_id: str) -> float:
        """K-factor applicable à ce joueur pour SON PROCHAIN match, calculé
        à partir de son nombre de matchs déjà joués (donc uniquement des
        matchs strictement antérieurs)."""
        if self.k_mode == "fixed":
            return self.k_factor
        m = self.matches_played[player_id]
        k = DYNAMIC_K_BASE / math.pow(m + DYNAMIC_K_OFFSET, DYNAMIC_K_EXPONENT)
        return max(DYNAMIC_K_MIN, min(DYNAMIC_K_MAX, k))

    @staticmethod
    def _apply_update(ratings: dict[str, float], winner_id: str, loser_id: str,
                       k_winner: float, k_loser: float) -> None:
        w_rating = ratings[winner_id]
        l_rating = ratings[loser_id]
        expected_w = expected_score(w_rating, l_rating)
        expected_l = 1.0 - expected_w
        ratings[winner_id] = w_rating + k_winner * (1.0 - expected_w)
        ratings[loser_id] = l_rating + k_loser * (0.0 - expected_l)

    def update(self, winner_id: str, loser_id: str, surface: str,
               indoor_label: str, decisive: bool) -> None:
        k_winner = self.k_for(winner_id)
        k_loser = self.k_for(loser_id)

        # --- Elo global ---
        # NB: en mode dynamique, chaque joueur applique SON PROPRE K à sa
        # propre mise à jour (convention 538) — ce n'est pas un K partagé
        # unique pour le match, ce qui permet à un débutant de gagner/perdre
        # plus de points qu'un joueur expérimenté sur le même match.
        self._apply_update(self.overall, winner_id, loser_id, k_winner, k_loser)

        # --- Elo par surface (si la surface est suivie) ---
        if surface in self.by_surface:
            self._apply_update(self.by_surface[surface], winner_id, loser_id, k_winner, k_loser)

        # --- Elo par environnement (surface + indoor/outdoor) ---
        if self.split_indoor and surface in TRACKED_SURFACES and indoor_label != "Unknown":
            key = f"{surface}-{indoor_label}"
            self._apply_update(self.by_surface_env[key], winner_id, loser_id, k_winner, k_loser)

        # --- Clutch Elo (uniquement si le match est allé au bout) ---
        if self.clutch_elo_enabled and decisive:
            self._apply_update(self.clutch, winner_id, loser_id, k_winner, k_loser)

        self.matches_played[winner_id] += 1
        self.matches_played[loser_id] += 1


def load_matches() -> list[dict]:
    with INPUT_FILE.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def process(matches: list[dict], k_factor: float, k_mode: str,
            split_indoor: bool, clutch_elo: bool) -> tuple[list[dict], EloTracker, dict]:
    """
    Parcourt les matchs dans l'ordre (déjà trié en amont par clean_data.py)
    et calcule, pour chaque match, les ratings PRE-match des deux joueurs
    avant de les mettre à jour. Renvoie :
      - la liste des lignes enrichies des colonnes Elo,
      - le tracker final (pour extraire le classement Elo courant),
      - des statistiques de contrôle (matchs ignorés, etc.)
    """
    tracker = EloTracker(k_factor=k_factor, k_mode=k_mode,
                          split_indoor=split_indoor, clutch_elo=clutch_elo)
    enriched_rows: list[dict] = []
    stats = {"skipped_missing_id": 0, "surface_untracked": 0, "n_matches": 0,
              "n_decisive_matches": 0, "n_indoor_unknown": 0}

    for row in matches:
        winner_id = (row.get("winner_id") or "").strip()
        loser_id = (row.get("loser_id") or "").strip()
        surface = (row.get("surface_norm") or "").strip()
        indoor_label = normalize_indoor(row.get("indoor", ""))
        decisive = is_decisive_match(row.get("score", ""), row.get("best_of", ""))

        if indoor_label == "Unknown":
            stats["n_indoor_unknown"] += 1
        if decisive:
            stats["n_decisive_matches"] += 1

        if not winner_id or not loser_id:
            stats["skipped_missing_id"] += 1
            # On garde quand même la ligne dans la sortie (traçabilité),
            # mais sans ratings ni mise à jour Elo pour ce match.
            new_row = dict(row)
            new_row.update({
                "winner_elo_pre": "", "loser_elo_pre": "",
                "winner_surface_elo_pre": "", "loser_surface_elo_pre": "",
                "winner_surface_env_elo_pre": "", "loser_surface_env_elo_pre": "",
                "winner_clutch_elo_pre": "", "loser_clutch_elo_pre": "",
                "elo_prob_winner": "",
            })
            enriched_rows.append(new_row)
            continue

        if surface not in TRACKED_SURFACES:
            stats["surface_untracked"] += 1

        # 1) Lecture des ratings PRE-match (état construit uniquement à
        #    partir des matchs strictement antérieurs déjà traités).
        winner_elo_pre = tracker.get_overall(winner_id)
        loser_elo_pre = tracker.get_overall(loser_id)
        winner_surface_elo_pre = tracker.get_surface(winner_id, surface)
        loser_surface_elo_pre = tracker.get_surface(loser_id, surface)
        winner_surface_env_elo_pre = tracker.get_surface_env(winner_id, surface, indoor_label)
        loser_surface_env_elo_pre = tracker.get_surface_env(loser_id, surface, indoor_label)
        winner_clutch_elo_pre = tracker.get_clutch(winner_id)
        loser_clutch_elo_pre = tracker.get_clutch(loser_id)

        elo_prob_winner = expected_score(winner_elo_pre, loser_elo_pre)

        new_row = dict(row)
        new_row["winner_elo_pre"] = round(winner_elo_pre, 2)
        new_row["loser_elo_pre"] = round(loser_elo_pre, 2)
        new_row["winner_surface_elo_pre"] = (
            round(winner_surface_elo_pre, 2) if winner_surface_elo_pre is not None else ""
        )
        new_row["loser_surface_elo_pre"] = (
            round(loser_surface_elo_pre, 2) if loser_surface_elo_pre is not None else ""
        )
        new_row["winner_surface_env_elo_pre"] = (
            round(winner_surface_env_elo_pre, 2) if winner_surface_env_elo_pre is not None else ""
        )
        new_row["loser_surface_env_elo_pre"] = (
            round(loser_surface_env_elo_pre, 2) if loser_surface_env_elo_pre is not None else ""
        )
        new_row["winner_clutch_elo_pre"] = (
            round(winner_clutch_elo_pre, 2) if winner_clutch_elo_pre is not None else ""
        )
        new_row["loser_clutch_elo_pre"] = (
            round(loser_clutch_elo_pre, 2) if loser_clutch_elo_pre is not None else ""
        )
        new_row["elo_prob_winner"] = round(elo_prob_winner, 4)
        enriched_rows.append(new_row)

        # 2) Mise à jour POST-match (ne doit jamais influencer les colonnes
        #    ci-dessus, qui sont figées avant cet appel).
        tracker.update(winner_id, loser_id, surface, indoor_label, decisive)
        stats["n_matches"] += 1

    return enriched_rows, tracker, stats


def write_matches_output(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ratings_output(tracker: EloTracker, matches: list[dict], output_path: Path) -> None:
    """Écrit le classement Elo final par joueur (overall + par surface),
    utile pour validation manuelle (comparaison à des Elo publiés connus)."""
    # On récupère le dernier nom connu pour chaque player_id, pour lisibilité.
    last_name: dict[str, str] = {}
    for row in matches:
        wid, lid = (row.get("winner_id") or "").strip(), (row.get("loser_id") or "").strip()
        if wid:
            last_name[wid] = row.get("winner_name", "") or last_name.get(wid, "")
        if lid:
            last_name[lid] = row.get("loser_name", "") or last_name.get(lid, "")

    all_ids = set(tracker.overall.keys())
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        env_keys = sorted(tracker.by_surface_env.keys()) if tracker.split_indoor else []
        fieldnames = (
            ["player_id", "player_name", "matches_played", "elo_overall"]
            + [f"elo_{s.lower()}" for s in TRACKED_SURFACES]
            + [f"elo_{k.lower().replace('-', '_')}" for k in env_keys]
            + (["elo_clutch"] if tracker.clutch_elo_enabled else [])
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pid in sorted(all_ids, key=lambda p: -tracker.overall[p]):
            row = {
                "player_id": pid,
                "player_name": last_name.get(pid, ""),
                "matches_played": tracker.matches_played[pid],
                "elo_overall": round(tracker.overall[pid], 1),
            }
            for surface in TRACKED_SURFACES:
                row[f"elo_{surface.lower()}"] = round(tracker.by_surface[surface][pid], 1)
            for k in env_keys:
                row[f"elo_{k.lower().replace('-', '_')}"] = round(tracker.by_surface_env[k][pid], 1)
            if tracker.clutch_elo_enabled:
                row["elo_clutch"] = round(tracker.clutch[pid], 1)
            writer.writerow(row)


def write_report(tracker: EloTracker, stats: dict, k_factor: float, k_mode: str, output_path: Path) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Top 15 Elo global et top 10 par surface, pour un contrôle visuel rapide
    # (Étape 3: "Valider le calcul en comparant à des Elo déjà publiés").
    top_overall = sorted(tracker.overall.items(), key=lambda kv: -kv[1])[:15]

    k_desc = (f"{k_factor} (fixe)" if k_mode == "fixed"
              else f"dynamique — K={DYNAMIC_K_BASE}/(m+{DYNAMIC_K_OFFSET})^{DYNAMIC_K_EXPONENT}, "
                   f"borné [{DYNAMIC_K_MIN}, {DYNAMIC_K_MAX}]")

    lines = [
        "# Rapport — build_elo.py",
        f"\nGénéré le {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC\n",
        f"**Mode K-factor :** {k_desc}",
        f"**Matchs traités pour la mise à jour Elo :** {stats['n_matches']}",
        f"**Matchs ignorés (ID joueur manquant) :** {stats['skipped_missing_id']}",
        f"**Matchs sur surface non suivie (ex: Unknown) — Elo global mis à jour, "
        f"Elo par surface non mis à jour :** {stats['surface_untracked']}",
        f"**Matchs avec environnement indoor/outdoor inconnu :** {stats['n_indoor_unknown']}",
        f"**Matchs 'allés au bout' (set décisif joué, comptent pour le Clutch Elo) :** "
        f"{stats['n_decisive_matches']} ({round(100*stats['n_decisive_matches']/max(1,stats['n_matches']),1)}%)\n",
        "## Top 15 — Elo global (fin de période)",
        "| Rang | Joueur (ID) | Elo |",
        "|---|---|---|",
    ]
    for rank, (pid, elo) in enumerate(top_overall, start=1):
        lines.append(f"| {rank} | player_id={pid} | {elo:.1f} |")

    lines.append(
        "\n⚠️ Les noms ne sont pas repris ici — voir le fichier `elo_ratings_final*.csv` "
        "correspondant pour le classement complet avec `player_name`, à utiliser pour la "
        "validation manuelle contre des Elo publiés (ex. Tennis Abstract)."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ Rapport écrit : {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcule l'Elo global, par surface, par environnement, et clutch.")
    parser.add_argument("--k-factor", type=float, default=DEFAULT_K,
                         help=f"K-factor fixe pour la mise à jour Elo (défaut: {DEFAULT_K}). "
                              f"Ignoré si --k-mode=dynamic.")
    parser.add_argument("--k-mode", choices=["fixed", "dynamic"], default="dynamic",
                         help="'dynamic' (défaut) : K décroît avec l'expérience du joueur "
                              f"(K={DYNAMIC_K_BASE}/(m+{DYNAMIC_K_OFFSET})^{DYNAMIC_K_EXPONENT}, "
                              f"borné entre {DYNAMIC_K_MIN} et {DYNAMIC_K_MAX}). "
                              "'fixed' : K constant pour tous les joueurs.")
    parser.add_argument("--no-split-indoor", dest="split_indoor", action="store_false",
                         help="Désactive l'Elo par environnement (surface + indoor/outdoor).")
    parser.add_argument("--no-clutch-elo", dest="clutch_elo", action="store_false",
                         help="Désactive le Clutch Elo (matchs allés au set décisif).")
    parser.set_defaults(split_indoor=True, clutch_elo=True)
    parser.add_argument("--suffix", type=str, default="",
                         help="Suffixe ajouté aux fichiers de sortie (ex: '_dynamic') "
                              "pour comparer plusieurs runs sans s'écraser.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output_matches_file = PROCESSED_DIR / f"atp_matches_with_elo{args.suffix}.csv"
    output_ratings_file = OUTPUTS_DIR / f"elo_ratings_final{args.suffix}.csv"
    output_report_file = OUTPUTS_DIR / f"build_elo_report{args.suffix}.md"

    print("=" * 70)
    print(f"ÉTAPE 3 — Calcul de l'Elo (global + par surface) — mode K: {args.k_mode}")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"✗ {INPUT_FILE} introuvable. Lance d'abord scripts/clean_data.py.")
        return 1

    matches = load_matches()
    print(f"→ {len(matches)} matchs chargés depuis {INPUT_FILE.name} (ordre supposé déjà chronologique).")

    enriched_rows, tracker, stats = process(matches, k_factor=args.k_factor, k_mode=args.k_mode,
                                             split_indoor=args.split_indoor, clutch_elo=args.clutch_elo)
    print(f"→ Elo mis à jour sur {stats['n_matches']} matchs "
          f"({stats['skipped_missing_id']} ignorés pour ID manquant, "
          f"{stats['surface_untracked']} sur surface non suivie, "
          f"{stats['n_decisive_matches']} allés au set décisif).")

    write_matches_output(enriched_rows, output_matches_file)
    print(f"→ Fichier enrichi écrit : {output_matches_file}")

    write_ratings_output(tracker, matches, output_ratings_file)
    print(f"→ Classement Elo final écrit : {output_ratings_file}")

    write_report(tracker, stats, args.k_factor, args.k_mode, output_report_file)


    return 0


if __name__ == "__main__":
    sys.exit(main())