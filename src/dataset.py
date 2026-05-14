import torch
from torch.utils.data import Dataset


class NepaliEnglishDataset(Dataset):
    def __init__(self, df, ne_tokenizer, en_tokenizer, max_len=128):
        self.df = df
        self.ne_tokenizer = ne_tokenizer
        self.en_tokenizer = en_tokenizer
        self.max_len = max_len

        # Source (Nepali) special token IDs
        self.src_pad_id = ne_tokenizer.token_to_id("[PAD]")
        self.src_sos_id = ne_tokenizer.token_to_id("[SOS]")
        self.src_eos_id = ne_tokenizer.token_to_id("[EOS]")

        # Target (English) special token IDs
        self.trg_pad_id = en_tokenizer.token_to_id("[PAD]")
        self.trg_sos_id = en_tokenizer.token_to_id("[SOS]")
        self.trg_eos_id = en_tokenizer.token_to_id("[EOS]")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get raw text
        src_text = str(self.df.iloc[idx]['nepali_sent'])
        trg_text = str(self.df.iloc[idx]['english_sent'])

        # 2. Encode to IDs and add [SOS] and [EOS]
        src_ids = [self.src_sos_id] + self.ne_tokenizer.encode(src_text).ids + [self.src_eos_id]
        trg_ids = [self.trg_sos_id] + self.en_tokenizer.encode(trg_text).ids + [self.trg_eos_id]

        # 3. Pad or truncate
        def pad_or_truncate(ids, pad_id):
            if len(ids) > self.max_len:
                return ids[:self.max_len]
            return ids + [pad_id] * (self.max_len - len(ids))

        return {
            "src": torch.tensor(pad_or_truncate(src_ids, self.src_pad_id), dtype=torch.long),
            "trg": torch.tensor(pad_or_truncate(trg_ids, self.trg_pad_id), dtype=torch.long),
        }