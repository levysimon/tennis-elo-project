"""
build_dataset.py
=================
Assemblage du dataset de modélisation.

Deux opérations critiques ici :

1. SYMÉTRISATION (éviter la fuite la plus sournoise du projet)
   -------------------------------------------------------------
   Jusqu'ici, toutes les features sont organisées en colonnes
   "winner_*"/"w_*" vs "loser_*"/"l_*". Si on entraînait un modèle
   directement là-dessus, il apprendrait trivialement "le camp gagnant a
   toujours plus de victoires H2H / un Elo légèrement plus élevé" — ce qui
   n'est pas un pattern prédictif mais un artefact de mise en forme : la
   colonne "winner_elo_pre" est SYSTÉMATIQUEMENT associée à la victoire par
   construction, peu importe le contenu réel des données.

   On corrige ça en ré-étiquetant chaque match "joueur_1 vs joueur_2" où
   l'assignation gagnant->joueur_1 ou gagnant->joueur_2 est déterministe
   MAIS indépendante du résultat (hash stable du match, pas un vrai
   mélange aléatoire différent à chaque run — on veut la reproductibilité).
   La cible devient `target_player1_won` (1 ou 0), et le modèle doit
   apprendre à partir des features des deux joueurs, plus symétriquement.

2. SPLIT TEMPOREL (jamais aléatoire, cf. principe méthodologique #2)
   -------------------------------------------------------------
   Toutes les lignes avant `--cutoff-date` (défaut: 2025-01-01) vont en
   train, toutes celles à partir de cette date vont en test. Un split
   aléatoire mélangerait passé et futur et fausserait complètement
   l'évaluation.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --cutoff-date 2025-01-01
    python scripts/build_dataset.py --input atp_matches_with_h2h.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT/ "outputs"

DEFAULT_INPUT_FILE = PROCESSED_DIR / "atp_matches_with_h2h.csv"
OUTPUT_FULL_FILE = PROCESSED_DIR / "atp_modeling_dataset.csv"   
OUTPUT_REPORT_FILE = OUTPUTS_DIR / "build_dataset_report.md"

# Paires de préfixes utilisées par les scripts précédents pour distinguer
# gagnant/perdant. Une colonne "winner_xxx" est symétrisée avec son pendant
# "loser_xxx" (même chose pour "w_xxx" / "l_xxx"). `elo_prob_winner` est un
# cas particulier (une seule colonne, pas de pendant) : on la recalcule
# proprement du point de vue de player_1 après symétrisation.
# Paires basées sur des préfixes (ex: winner_rank / loser_rank -> player1_rank / player2_rank)
PREFIX_PAIRS = [("winner_", "loser_"), ("w_", "l_")]

# Paires exactes pour les cotes (ex: B365W / B365L -> player1_b365 / player2_b365)
EXACT_ODDS_PAIRS = [
    ("B365W", "B365L", "b365"),
    ("PSW", "PSL", "ps"),
    ("EXW", "EXL", "ex"),
    ("MaxW", "MaxL", "max"),
    ("AvgW", "AvgL", "avg"),
]
SPECIAL_RECOMPUTED_COLUMNS = {"elo_prob_winner"}


def classify_tier(tourney_level: str, draw_size: str) -> str:
    """Classification APPROXIMATIVE du niveau de tournoi, à partir du code
    `tourney_level` (convention Sackmann/TennisMyLife : G=Grand Chelem,
    M=Masters 1000, F=Masters de fin d'année, D=Coupe Davis, A=reste du
    circuit ATP) et, pour les tournois de catégorie 'A', d'un seuil sur
    `draw_size` pour approximer ATP 500 vs ATP 250.

    ⚠️ Cette distinction 500/250 par taille de tableau est une HEURISTIQUE :
    les tailles de tableau se chevauchent partiellement entre catégories
    (certains 250 ont un tableau de 32, comme certains 500). Pour une
    classification exacte, il faudrait une table de correspondance
    tournoi->catégorie tenue à jour séparément. À affiner si cette feature
    s'avère importante pour le modèle (cf. étape 8, importance des features)."""
    level = (tourney_level or "").strip().upper()
    if level == "G":
        return "Grand Slam"
    if level == "M":
        return "Masters 1000"
    if level == "F":
        return "Tour Finals"
    if level == "250":
        return "ATP 250"
    if level == "500":
        return "ATP 500"
    if level == "D":
        return "Davis Cup"
    if level == "A":
        try:
            size = int(float(draw_size)) if draw_size not in (None, "") else 0
        except ValueError:
            size = 0
        return "ATP 500" if size >= 48 else "ATP 250"
    return "Other"


def deterministic_bit(seed_string: str) -> int:
    """Renvoie 0 ou 1 de façon déterministe et reproductible à partir d'une
    chaîne (ex: identifiant de match), SANS dépendre du résultat du match.
    Utilise MD5 plutôt que hash() built-in, qui est aléatoire d'un run
    Python à l'autre pour les chaînes (PYTHONHASHSEED)."""
    digest = hashlib.md5(seed_string.encode("utf-8")).hexdigest()
    return int(digest[0], 16) % 2


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def symmetrize_row(row: dict, player1_is_winner: bool) -> dict:
    handled_keys: set[str] = set(SPECIAL_RECOMPUTED_COLUMNS)
    player1: dict = {}
    player2: dict = {}
    shared: dict = {}

    # A. Traitement des cotes (Paires exactes)
    for w_col, l_col, name in EXACT_ODDS_PAIRS:
        if w_col in row and l_col in row:
            handled_keys.add(w_col)
            handled_keys.add(l_col)
            w_val, l_val = row[w_col], row[l_col]
            if player1_is_winner:
                player1[name], player2[name] = w_val, l_val
            else:
                player1[name], player2[name] = l_val, w_val

    # B. Traitement des features préfixées (winner_* / loser_*)
    for key, value in row.items():
        if key in handled_keys:
            continue
        matched = False
        for prefix_w, prefix_l in PREFIX_PAIRS:
            if key.startswith(prefix_w):
                base = key[len(prefix_w):]
                loser_key = prefix_l + base
                if loser_key in row:
                    handled_keys.add(key)
                    handled_keys.add(loser_key)
                    w_val, l_val = value, row[loser_key]
                    if player1_is_winner:
                        player1[base], player2[base] = w_val, l_val
                    else:
                        player1[base], player2[base] = l_val, w_val
                    matched = True
                break
        if matched:
            continue

    # C. Traitement des colonnes partagées
    for key, value in row.items():
        if key not in handled_keys:
            shared[key] = value

    out = {f"player1_{k}": v for k, v in player1.items()}
    out.update({f"player2_{k}": v for k, v in player2.items()})
    out.update(shared)

    # Recalcul propre de la probabilité Elo pre-match du point de vue de
    # player_1, à partir des ratings déjà symétrisés
    elo1 = out.get("player1_elo_pre", "")
    elo2 = out.get("player2_elo_pre", "")
    if elo1 not in ("", None) and elo2 not in ("", None):
        out["player1_elo_win_prob"] = round(expected_score(float(elo1), float(elo2)), 4)
    else:
        out["player1_elo_win_prob"] = ""

    out["target_player1_won"] = 1 if player1_is_winner else 0

    # --- Features contextuelles (idées du 31/08) ---
    # `best_of`, `tourney_level`, `draw_size` sont des colonnes PARTAGÉES (ni
    # gagnant ni perdant), déjà présentes dans `out` via le bucket `shared`.

    # Niveau de tournoi : classification approximative (voir classify_tier).
    out["tourney_tier"] = classify_tier(out.get("tourney_level", ""), out.get("draw_size", ""))
    out["is_grand_slam"] = 1 if str(out.get("tourney_level", "")).strip().upper() == "G" else 0
    out["is_masters_1000"] = 1 if str(out.get("tourney_level", "")).strip().upper() == "M" else 0

    # Confrontation avec un gaucher : dérivé de la main du CAMP ADVERSE
    # (convention Sackmann/TML : 'L'=gaucher, 'R'=droitier, 'U'=inconnu).
    # Vide si la main de l'adversaire est inconnue, plutôt qu'un faux 0.
    # `vs_lefty_win_pct` (taux de victoire historique contre des gauchers)
    # nécessiterait un tracker dédié façon build_h2h.py — non implémenté ici ;
    # ce flag ponctuel permet déjà de capter un effet de matchup sans historique.
    p1_hand = (player1.get("hand") or "").strip().upper()
    p2_hand = (player2.get("hand") or "").strip().upper()
    out["player1_faces_lefty"] = (1 if p2_hand == "L" else 0) if p2_hand in ("L", "R") else ""
    out["player2_faces_lefty"] = (1 if p1_hand == "L" else 0) if p1_hand in ("L", "R") else ""

    surface_raw = str(out.get("surface_norm") or out.get("surface") or "").strip()
    for surf_name in ("Hard", "Clay", "Grass", "Carpet"):
        out[f"surface_{surf_name}"] = 1 if surface_raw == surf_name else 0

    return out


def process(matches: list[dict]) -> tuple[list[dict], dict]:
    enriched_rows: list[dict] = []
    stats_counter = {
        "n_matches": 0, 
        "n_skipped_missing_id": 0, 
        "n_skipped_missing_odds": 0,  # <-- NOUVEAU COMPTEUR
        "n_player1_wins": 0
    }

    for row in matches:
        winner_id = (row.get("winner_id") or "").strip()
        loser_id = (row.get("loser_id") or "").strip()
        tourney_id = (row.get("tourney_id") or "").strip()
        match_num = (row.get("match_num") or "").strip()

        if not winner_id or not loser_id:
            stats_counter["n_skipped_missing_id"] += 1
            continue

        # --- NOUVEAU : Exclusion si les cotes Pinnacle sont absentes ---
        psw = str(row.get("PSW") or "").strip()
        psl = str(row.get("PSL") or "").strip()
        if not psw or not psl:
            stats_counter["n_skipped_missing_odds"] += 1
            continue
        # ---------------------------------------------------------------

        seed = f"{tourney_id}_{match_num}_{winner_id}_{loser_id}"
        player1_is_winner = deterministic_bit(seed) == 0

        new_row = symmetrize_row(row, player1_is_winner)
        if player1_is_winner:
            stats_counter["n_player1_wins"] += 1

        stats_counter["n_matches"] += 1
        enriched_rows.append(new_row)

    return enriched_rows, stats_counter


def split_train_val_test(
    rows: list[dict], 
    val_cutoff: str, 
    test_cutoff: str
) -> tuple[list[dict], list[dict], list[dict]]:
    train, val, test = [], [], []
    
    for row in rows:
        date_str = row.get("tourney_date", "")
        
        # Sécurité si la date est incomplète
        if len(date_str) < 10:
            row["dataset_split"] = "train"
            train.append(row)
            continue

        if date_str >= test_cutoff:
            row["dataset_split"] = "test"
            test.append(row)
        elif date_str >= val_cutoff:
            row["dataset_split"] = "validation"
            val.append(row)
        else:
            row["dataset_split"] = "train"
            train.append(row)
            
    return train, val, test


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(all_rows: list[dict], train: list[dict], test: list[dict],
                  stats_counter: dict, cutoff_date: str, output_path: Path) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    n_total = len(all_rows)
    balance = round(100 * stats_counter["n_player1_wins"] / n_total, 2) if n_total else 0

    from collections import Counter
    tier_counts = Counter(r.get("tourney_tier", "Other") for r in all_rows)
    raw_level_counts = Counter(str(r.get("tourney_level", "")).strip() or "(vide)" for r in all_rows)
    n_bo5 = sum(1 for r in all_rows if r.get("is_best_of_5") == 1)
    n_lefty_known = sum(1 for r in all_rows if r.get("player1_faces_lefty") != "")
    n_lefty_matchups = sum(1 for r in all_rows if r.get("player1_faces_lefty") == 1 or r.get("player2_faces_lefty") == 1)

    train_dates = sorted(r["tourney_date"] for r in train if len(r.get("tourney_date", "")) == 10)
    test_dates = sorted(r["tourney_date"] for r in test if len(r.get("tourney_date", "")) == 10)

    lines = [
        "# Rapport — build_dataset.py",
        f"\nGénéré le {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC\n",
        f"**Matchs symétrisés :** {n_total}",
        f"**Matchs ignorés (ID joueur manquant) :** {stats_counter['n_skipped_missing_id']}",
        f"**Matchs ignorés (Cotes Pinnacle manquantes) :** {stats_counter['n_skipped_missing_odds']}", # <-- NOUVELLE LIGNE
        f"**Équilibre de l'assignation player_1/player_2 :** {balance}% des matchs ont "
        f"player_1 vainqueur (doit être proche de 50% — un fort écart indiquerait un bug "
        f"dans `deterministic_bit`, PAS une propriété réelle du tennis)\n",
        f"**Date de coupure train/test :** {cutoff_date}",
        f"**Train :** {len(train)} matchs "
        f"({train_dates[0] if train_dates else '?'} → {train_dates[-1] if train_dates else '?'})",
        f"**Test :** {len(test)} matchs "
        f"({test_dates[0] if test_dates else '?'} → {test_dates[-1] if test_dates else '?'})\n",
        f"**Matchs en 5 sets (`is_best_of_5`) :** {n_bo5} ({round(100*n_bo5/n_total,2)}%)",
        f"**Confrontations avec main adverse connue :** {n_lefty_known} / {n_total}",
        f"**dont impliquant un gaucher :** {n_lefty_matchups}\n",
        "## Répartition par niveau de tournoi (`tourney_tier`, classification approximative — voir `classify_tier`)",
        "| Tier | Matchs |",
        "|---|---|",
    ]
    for tier, count in tier_counts.most_common():
        lines.append(f"| {tier} | {count} |")

    lines += [
        "\n## Valeurs brutes de `tourney_level` observées (pour vérifier la classification ci-dessus)",
        "| Valeur brute | Matchs |",
        "|---|---|",
    ]
    for level, count in raw_level_counts.most_common():
        lines.append(f"| {level} | {count} |")

    lines += [
        "\n## Rappel de conception",
        "- Chaque match apparaît une seule fois, avec des colonnes `player1_*` / `player2_*` "
        "symétriques (ni l'un ni l'autre n'est systématiquement le gagnant).",
        "- `target_player1_won` : 1 si player_1 a gagné, 0 sinon — c'est la cible du modèle.",
        "- `player1_elo_win_prob` : probabilité de victoire de player_1 selon le seul Elo "
        "pre-match, recalculée après symétrisation — sert de **baseline** à l'étape 8.",
        "- Le split est strictement temporel (pas de mélange aléatoire) : toute ligne dont "
        f"`tourney_date >= {cutoff_date}` va en test, le reste en train.",
        "\n## Nouvelles colonnes contextuelles (31/08)",
        "- `tourney_tier` : Grand Slam / Masters 1000 / ATP 500 / ATP 250 / Tour Finals / "
        "Davis Cup / Other — la distinction 500 vs 250 est une HEURISTIQUE sur `draw_size`, "
        "pas une classification garantie exacte (voir `classify_tier` dans le code).",
        "- `is_grand_slam` : 1/0.",
        "- `is_best_of_5` : 1/0, vide si `best_of` est absent.",
        "- `player1_faces_lefty` / `player2_faces_lefty` : 1 si l'ADVERSAIRE est gaucher, "
        "vide si sa main n'est pas renseignée (pas un faux 0).",
        "\n⚠️ Ces nouvelles colonnes sont entraînées sur TOUT l'historique (pas de filtre par "
        "tier ici) — le filtre éventuel ATP500/1000/Grand Chelem, pour limiter le biais de "
        "motivation évoqué, sera appliqué au moment de l'ÉVALUATION (evaluate.py), pas ici, "
        "pour ne pas appauvrir l'historique Elo/forme récente des joueurs.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ Rapport écrit : {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble le dataset de modélisation final (symétrisé, splitté 3-way).")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_FILE),
                        help=f"Fichier d'entrée. Défaut : {DEFAULT_INPUT_FILE.name}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_absolute() and "/" not in args.input:
        input_path = PROCESSED_DIR / args.input

    print("ÉTAPE 7 — Assemblage du dataset de modélisation")

    if not input_path.exists():
        print(f"✗ {input_path} introuvable. Lance d'abord scripts/build_h2h.py.")
        return 1

    with input_path.open("r", encoding="utf-8", newline="") as f:
        matches = list(csv.DictReader(f))
    print(f"→ {len(matches)} matchs chargés depuis {input_path.name}.")

    enriched_rows, stats_counter = process(matches)
    # Remplacez les prints d'affichage des stats dans main() par ceux-ci :
    print(f"→ {stats_counter['n_matches']} matchs symétrisés conservés.")
    print(f"  - {stats_counter['n_skipped_missing_id']} ignorés pour ID manquant.")
    print(f"  - {stats_counter['n_skipped_missing_odds']} ignorés pour cotes Pinnacle manquantes.")
    
    print(f"→ Équilibre player_1 vainqueur : "
          f"{round(100 * stats_counter['n_player1_wins'] / max(1, stats_counter['n_matches']), 2)}% "
          f"(doit être proche de 50%).")


    write_csv(enriched_rows, OUTPUT_FULL_FILE)


    print(f"→ Fichier écrit : {OUTPUT_FULL_FILE.name}")


    return 0


if __name__ == "__main__":
    sys.exit(main())