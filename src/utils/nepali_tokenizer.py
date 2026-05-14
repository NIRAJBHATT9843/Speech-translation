import sys, os
sys.path.append(os.path.dirname(__file__))
from NepaliSyllables import valid_symbols

class NepaliTokenizer:
    def __init__(self, win_size=4):
        self.win_size = win_size

        special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        all_tokens = special_tokens + valid_symbols

        self.token_to_id_map = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_token_map = {idx: token for token, idx in self.token_to_id_map.items()}

        self.pad_id = self.token_to_id_map["[PAD]"]
        self.unk_id = self.token_to_id_map["[UNK]"]
        self.sos_id = self.token_to_id_map["[SOS]"]
        self.eos_id = self.token_to_id_map["[EOS]"]

    def tokenize(self, sentence):
        FT = []
        # Process word by word, ignore spaces
        for word in sentence.strip().split():
            T = list(word)
            current_win_pos = 0
            while current_win_pos < len(T):
                t_window = T[current_win_pos:current_win_pos + self.win_size]
                found = False
                for length in range(len(t_window), 0, -1):
                    potential_syllable = ''.join(t_window[:length])
                    if potential_syllable in self.token_to_id_map:
                        FT.append(potential_syllable)
                        current_win_pos += length
                        found = True
                        break
                if not found:
                    FT.append(T[current_win_pos])
                    current_win_pos += 1
        return FT

    def encode(self, sentence):
        """Returns object with .ids attribute to match HuggingFace interface"""
        tokens = self.tokenize(sentence)
        ids = [self.token_to_id_map.get(t, self.unk_id) for t in tokens]
        return _EncodingResult(ids)

    def decode(self, ids, skip_special_tokens=True):
        special_ids = {self.pad_id, self.unk_id, self.sos_id, self.eos_id}
        tokens = []
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            tokens.append(self.id_to_token_map.get(id, "[UNK]"))
        return ''.join(tokens)

    def token_to_id(self, token):
        return self.token_to_id_map.get(token, self.unk_id)

    def get_vocab_size(self):
        # Use max ID + 1 so embedding size covers all possible IDs (handles duplicate tokens in valid_symbols)
        if not self.token_to_id_map:
            return 0
        return max(self.token_to_id_map.values()) + 1


class _EncodingResult:
    """Mimics HuggingFace tokenizer encode() output so dataset.py works unchanged"""
    def __init__(self, ids):
        self.ids = ids