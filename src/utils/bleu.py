import os
import torch
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU

from src.utils.tokenizer import load_nepali_tokenizer, load_english_tokenizer
from src.utils.translate import translate


# ─────────────────────────────────────────────────────────────────────────────
# BLEU Evaluation using sacrebleu + beam search
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bleu_scores(model, df, ne_tokenizer, en_tokenizer, device,
                         name="Test", beam_width=5, max_len=128):
    model.eval()

    hypotheses = []
    references = []

    print(f"Evaluating {name} set ({len(df)} samples) with beam_width={beam_width}...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        source_text = str(row['nepali_sent'])
        target_text = str(row['english_sent'])

        try:
            predicted_text = translate(
                model, source_text,
                ne_tokenizer, en_tokenizer,
                device, max_len=max_len,
                beam_width=beam_width
            )
            hypotheses.append(predicted_text)
            references.append(target_text)
        except Exception as e:
            print(f"  Skipping sample due to error: {e}")
            continue

    if not hypotheses:
        print(f"No samples evaluated in {name} set.")
        return None

    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(hypotheses, [references])

    print(f"\n--- FINAL RESULTS FOR {name.upper()} ---")
    print(f"BLEU Score : {result.score:.2f}")
    print(f"Details    : {result}")

    return result.score


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.model.model import build_transformer

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

    ne_tokenizer = load_nepali_tokenizer()
    en_tokenizer = load_english_tokenizer()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    model = build_transformer(
        src_vocab_size=ne_tokenizer.get_vocab_size(),
        tgt_vocab_size=en_tokenizer.get_vocab_size(),
        src_seq_len=128,
        tgt_seq_len=128,
        d_model=512,
        N=4,
        h=8,
        dropout=0.3,   # match your training config
        d_ff=2048
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_path = os.path.join(PROJECT_ROOT, "models", "best_nepali_to_eng.pth")
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")

    model.eval()

    # Evaluate on test set
    evaluate_bleu_scores(
        model, test_df,
        ne_tokenizer, en_tokenizer,
        device, name="Test",
        beam_width=5
    )