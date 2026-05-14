
'''Using Beam Search for Translation'''

import re
import torch


def translate(model, sentence, ne_tokenizer, en_tokenizer, device,
              max_len=128, beam_width=5):
    """
    Translate a Nepali sentence to English.
    beam_width=1  → greedy search (old behavior)
    beam_width=5  → beam search (better quality, recommended)
    """

    model.eval()

    # ── Tokenize input ────────────────────────────────────────────────────────
    ne_sos = ne_tokenizer.token_to_id("[SOS]")
    ne_eos = ne_tokenizer.token_to_id("[EOS]")
    ne_pad = ne_tokenizer.token_to_id("[PAD]")

    tokens     = [ne_sos] + ne_tokenizer.encode(sentence).ids + [ne_eos]
    src_tensor = torch.tensor([tokens]).to(device)
    src_mask   = (src_tensor != ne_pad).unsqueeze(1).unsqueeze(2)

    # ── Encode once ───────────────────────────────────────────────────────────
    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)

    en_sos = en_tokenizer.token_to_id("[SOS]")
    en_eos = en_tokenizer.token_to_id("[EOS]")

    # ── Greedy search (beam_width=1) ──────────────────────────────────────────
    if beam_width == 1:
        decoder_input = torch.tensor([[en_sos]], device=device)

        for _ in range(max_len):
            size     = decoder_input.size(1)
            tgt_mask = torch.triu(torch.ones((1, 1, size, size)), diagonal=1).bool().to(device)
            tgt_mask = ~tgt_mask

            with torch.no_grad():
                out       = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
                prob      = model.project(out[:, -1])
                next_word = torch.argmax(prob, dim=1).item()

            if next_word == en_eos:
                break

            decoder_input = torch.cat(
                [decoder_input, torch.tensor([[next_word]], device=device)], dim=1
            )

        decoded_indices = decoder_input[0].tolist()

    # ── Beam search (beam_width > 1) ──────────────────────────────────────────
    else:
        beams     = [(0.0, [en_sos])]
        completed = []

        for _ in range(max_len):
            all_candidates = []

            for score, tokens_so_far in beams:
                if tokens_so_far[-1] == en_eos:
                    completed.append((score, tokens_so_far))
                    continue

                decoder_input = torch.tensor([tokens_so_far], device=device)
                size          = decoder_input.size(1)
                tgt_mask      = torch.triu(torch.ones((1, 1, size, size)), diagonal=1).bool().to(device)
                tgt_mask      = ~tgt_mask

                with torch.no_grad():
                    out   = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
                    probs = torch.log_softmax(model.project(out[:, -1]), dim=-1)

                top_probs, top_ids = probs[0].topk(beam_width)

                for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
                    candidate = (score + prob, tokens_so_far + [token_id])
                    all_candidates.append(candidate)

            if not all_candidates:
                break

            all_candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            beams = all_candidates[:beam_width]

            if all(t[-1] == en_eos for _, t in beams):
                completed.extend(beams)
                break

        if completed:
            completed.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            decoded_indices = completed[0][1]
        else:
            beams.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            decoded_indices = beams[0][1]

    # ── Decode tokens to string ───────────────────────────────────────────────
    special_tokens = {en_sos, en_eos, en_tokenizer.token_to_id("[PAD]")}
    final_indices  = [idx for idx in decoded_indices if idx not in special_tokens]

    # Merge subwords properly using Ġ marker
    tokens_list  = [en_tokenizer.id_to_token(i) for i in final_indices]
    words        = []
    current_word = ""

    for token in tokens_list:
        if token is None:
            continue
        if token.startswith('Ġ'):
            # New word — save current and start new
            if current_word:
                words.append(current_word)
            current_word = token[1:]   # remove Ġ prefix
        else:
            # Subword continuation — merge without space
            current_word += token

    if current_word:
        words.append(current_word)

    clean_output = ' '.join(words)

    # Fix contractions
    clean_output = clean_output.replace(" 't",  "'t")    # don 't  → don't
    clean_output = clean_output.replace(" n't", "n't")   # can n't → can't
    clean_output = clean_output.replace(" 's",  "'s")    # he 's   → he's
    clean_output = clean_output.replace(" 're", "'re")   # they 're→ they're
    clean_output = clean_output.replace(" 've", "'ve")   # I 've   → I've
    clean_output = clean_output.replace(" 'll", "'ll")   # I 'll   → I'll
    clean_output = clean_output.replace(" 'd",  "'d")    # I 'd    → I'd
    clean_output = clean_output.replace(" 'm",  "'m")    # I 'm    → I'm

    # Fix punctuation spacing
    clean_output = clean_output.replace(" ,", ",")
    clean_output = clean_output.replace(" .", ".")
    clean_output = clean_output.replace(" ?", "?")
    clean_output = clean_output.replace(" !", "!")
    clean_output = clean_output.replace(" ;", ";")
    clean_output = clean_output.replace(" :", ":")

    # Fix multiple spaces
    clean_output = re.sub(r'\s+', ' ', clean_output).strip()

    # Capitalize first letter of sentence
    if clean_output:
        clean_output = clean_output[0].upper() + clean_output[1:]

    return clean_output
