"""GRPO multi-tour + recombinaison SNIS des tours — point d'entrée.

Mêmes arguments que train_grpo.py (parser partagé, --help les montre tous),
plus le groupe SNIS :
    --snis-alpha 1.0             exposant α sur les poids SNIS
    --snis-real-rollouts 0       G = épisodes réels joués par prompt ; 0 = M
                                 (pas de découplage : autant de combos que de rollouts)
    --snis-keep-originals 0      nb de trajectoires originales gardées telles
                                 quelles parmi les M sorties (0 = SNIS pur, <= G)
    --snis-score-batch-size 4    batch du forward HF de scoring
    --snis-max-ctx-tokens 15000  budget tokens du contexte d'un combo
    --snis-seed 0                graine du tirage ancestral (déterministe par step)
    --selftest                   test CPU de la logique de recombinaison (sans GPU,
                                 sans serveur TextCraft) puis exit

La méthode (maths, approximations v1, diagnostics ESS) est documentée dans
src/train/snis.py et docs/hebdo/10juillet/snis_derivation.tex. Ce fichier ne
fait qu'assembler : il construit une SnisConfig depuis la CLI et délègue à
train_grpo.main(rollout_func=snis.make_rollout_func(cfg)) — la loss GRPO de
TRL reste strictement inchangée, seule la rollout_func diffère.

Exemple (M=32 combos par prompt, G=8 rollouts réels, soit 4x plus de données de
gradient que GRPO classique pour le même coût d'environnement) :
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python src/train/train_grpo_snis.py --full-ft --num-generations 32 \
        --snis-real-rollouts 8 --gradient-accumulation-steps 256 \
        --max-completion-length 512 --max-items 0 --max-steps 200 \
        --use-vllm-inprocess --snis-alpha 1.0 --run-name exp18_snis_v2
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du repo → imports src.*

from src.train import snis, train_grpo


def main() -> None:
    parser = train_grpo.build_parser()
    group = parser.add_argument_group("snis", "Recombinaison SNIS des tours (voir src/train/snis.py)")
    group.add_argument("--snis-alpha", type=float, default=1.0)
    group.add_argument("--snis-real-rollouts", type=int, default=0)
    group.add_argument("--snis-keep-originals", type=int, default=0)
    group.add_argument("--snis-score-batch-size", type=int, default=4)
    group.add_argument("--snis-max-ctx-tokens", type=int, default=15000)
    group.add_argument("--snis-seed", type=int, default=0)
    group.add_argument("--selftest", action="store_true",
                       help="Valide la logique de recombinaison sur CPU puis exit.")
    args = parser.parse_args()

    if args.selftest:
        snis.run_selftest()
        return

    cfg = snis.SnisConfig(
        alpha=args.snis_alpha,
        real_rollouts=args.snis_real_rollouts,
        keep_originals=args.snis_keep_originals,
        score_batch_size=args.snis_score_batch_size,
        max_ctx_tokens=args.snis_max_ctx_tokens,
        seed=args.snis_seed,
    )
    print(f"[snis] α={cfg.alpha} real_rollouts_G={cfg.real_rollouts or 'M (=num-generations)'} "
          f"keep_originals={cfg.keep_originals} "
          f"score_batch={cfg.score_batch_size} max_ctx={cfg.max_ctx_tokens} seed={cfg.seed}", flush=True)

    train_grpo.main(args, rollout_func=snis.make_rollout_func(cfg))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
