"""
Vérifie que les transcripts d'eval n'ont pas dépassé la fenêtre de contexte de Qwen.

Pour chaque épisode JSON, reconstruit la conversation tour par tour et mesure la
longueur en tokens à chaque round. Signale tout épisode ayant dépassé max_context.

Usage :
    python src/analysis/check_context_length.py \
        --eval-dir runs/exp7.1_ckpt1598/eval_logs \
        --tokenizer saves/trl_grpo/exp7.1_b200_fullft_12ep/checkpoint-1598

    # Ou depuis le modèle de base (tokenizer identique) :
    python src/analysis/check_context_length.py \
        --eval-dir runs/exp7.1_ckpt1598/eval_logs \
        --tokenizer models/Qwen2.5-3B-Instruct
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


MAX_CONTEXT = 32768  # Qwen2.5-3B-Instruct context window


def count_tokens(tokenizer, messages: list[dict]) -> int:
    """Tokens de la conversation complète après application du chat template."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def analyze_episode(tokenizer, log_path: Path) -> dict:
    with log_path.open() as f:
        data = json.load(f)

    transcript = data.get("transcript", [])
    item_id = data["item_id"]
    reward = data["reward"]

    # Reconstruit la conversation tour par tour et mesure les tokens
    token_counts = []
    messages = []
    for msg in transcript:
        messages.append(msg)
        # On mesure après chaque message "user" (= fin d'un tour complet)
        if msg["role"] == "user":
            n = count_tokens(tokenizer, messages)
            token_counts.append(n)

    max_tokens = max(token_counts) if token_counts else 0
    exceeded = max_tokens > MAX_CONTEXT

    return {
        "item_id": item_id,
        "reward": reward,
        "rounds": data["rounds"],
        "max_tokens": max_tokens,
        "exceeded": exceeded,
        "token_counts_per_round": token_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, help="Dossier contenant les JSON d'eval")
    parser.add_argument("--tokenizer", required=True, help="Chemin vers le tokenizer (checkpoint ou modèle de base)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    logs = sorted(eval_dir.glob("textcraft_*.json"))
    if not logs:
        print(f"Aucun log trouvé dans {eval_dir}")
        return

    print(f"[check] Chargement du tokenizer depuis {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"[check] {len(logs)} épisodes à analyser. Fenêtre de contexte : {MAX_CONTEXT} tokens.\n")

    results = []
    for i, log_path in enumerate(logs):
        r = analyze_episode(tokenizer, log_path)
        results.append(r)
        status = "⚠️  DÉPASSÉ" if r["exceeded"] else "OK"
        print(f"[{i+1:3d}/{len(logs)}] {r['item_id']}  rounds={r['rounds']}  "
              f"max_tokens={r['max_tokens']:6d}  reward={r['reward']:.0f}  {status}")

    max_tokens_all = max(r["max_tokens"] for r in results)
    mean_tokens = sum(r["max_tokens"] for r in results) / len(results)
    exceeded = [r for r in results if r["exceeded"]]
    over_half = [r for r in results if r["max_tokens"] > MAX_CONTEXT // 2]

    print("\n" + "=" * 60)
    print(f"RÉSUMÉ — {len(logs)} épisodes")
    print(f"  Max tokens (pire cas)   : {max_tokens_all:,}  / {MAX_CONTEXT:,}")
    print(f"  Moyenne max tokens      : {mean_tokens:,.0f}")
    print(f"  Épisodes > 50% contexte : {len(over_half)}")
    print(f"  Épisodes DÉPASSÉS       : {len(exceeded)}")
    if exceeded:
        print("\n  ⚠️  Épisodes ayant dépassé la fenêtre :")
        for r in exceeded:
            print(f"     {r['item_id']}  max={r['max_tokens']}  rounds={r['rounds']}")
    else:
        print("\n  ✅ Aucun épisode n'a dépassé la fenêtre de contexte.")
    print("=" * 60)


if __name__ == "__main__":
    main()
