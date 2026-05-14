# import torch

# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())


# from src.utils.tokenizer import load_nepali_tokenizer, load_english_tokenizer

# # Load tokenizers
# ne_tok = load_nepali_tokenizer()
# en_tok = load_english_tokenizer()

# # Test sentences
# ne_sentence = "म तिमीलाई माया गर्छु"
# en_sentence = "I love you"

# # Encode
# ne_encoded = ne_tok.encode(ne_sentence)
# en_encoded = en_tok.encode(en_sentence)

# print("✅ Nepali tokens:", ne_encoded.tokens)
# print("✅ Nepali IDs:", ne_encoded.ids)

# print("✅ English tokens:", en_encoded.tokens)
# print("✅ English IDs:", en_encoded.ids)


# # test.py
# import torch
# from src.utils.nepali_tokenizer import NepaliTokenizer
# from src.utils.tokenizer import load_english_tokenizer

# print("=" * 50)
# print("STEP 1: Testing Nepali Tokenizer")
# print("=" * 50)

# ne_tokenizer = NepaliTokenizer()
# print(f"Vocab size: {ne_tokenizer.get_vocab_size()}")
# print(f"PAD ID: {ne_tokenizer.token_to_id('[PAD]')}")
# print(f"SOS ID: {ne_tokenizer.token_to_id('[SOS]')}")
# print(f"EOS ID: {ne_tokenizer.token_to_id('[EOS]')}")

# sentence = "खाना"
# tokens = ne_tokenizer.tokenize(sentence)
# ids = ne_tokenizer.encode(sentence).ids
# decoded = ne_tokenizer.decode(ids)

# print(f"Original:  {sentence}")
# print(f"Tokens:    {tokens}")
# print(f"IDs:       {ids}")
# print(f"Decoded:   {decoded}")

# print("\n" + "=" * 50)
# print("STEP 2: Testing English Tokenizer")
# print("=" * 50)

# en_tokenizer = load_english_tokenizer()
# print(f"Vocab size: {en_tokenizer.get_vocab_size()}")
# print(f"PAD ID: {en_tokenizer.token_to_id('[PAD]')}")

# en_sentence = "food."
# en_ids = en_tokenizer.encode(en_sentence).ids
# print(f"Original: {en_sentence}")
# print(f"IDs:      {en_ids}")
# print(f"Decoded:  {en_tokenizer.decode(en_ids)}")

# print("\n" + "=" * 50)
# print("STEP 3: Testing Dataset")
# print("=" * 50)

# import pandas as pd
# from src.dataset import NepaliEnglishDataset

# # Create a tiny dummy dataframe to test
# dummy_df = pd.DataFrame({
#     'nepali_sent': ['खाना '],
#     'english_sent': ['food.']
# })

# dataset = NepaliEnglishDataset(dummy_df, ne_tokenizer, en_tokenizer)
# sample = dataset[0]
# print(f"Source shape: {sample['src'].shape}")
# print(f"Target shape: {sample['trg'].shape}")
# print(f"Source IDs: {sample['src'][:10]}...")
# print(f"Target IDs: {sample['trg'][:10]}...")

# print("\n" + "=" * 50)
# print("STEP 4: Testing Model Build")
# print("=" * 50)

# from src.model.model import build_transformer

# model = build_transformer(
#     src_vocab_size=ne_tokenizer.get_vocab_size(),
#     tgt_vocab_size=en_tokenizer.get_vocab_size(),
#     src_seq_len=64,
#     tgt_seq_len=64,
#     d_model=256,
#     N=3,
#     h=8,
#     dropout=0.1
# )
# print(f"Model built successfully!")
# print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# print("\n" + "=" * 50)
# print("ALL TESTS PASSED!")
# print("=" * 50)



# # Special Token IDs:
# #   [PAD]  | Nepali ID: 0 | English ID: 0
# #   [SOS]  | Nepali ID: 2 | English ID: 1
# #   [EOS]  | Nepali ID: 3 | English ID: 2
# #   [UNK]  | Nepali ID: 1 | English ID: 3


# # Run this to check your data
# import pandas as pd
# from src.utils.tokenizer import load_nepali_tokenizer, load_english_tokenizer
# from src.utils.nepali_tokenizer import NepaliTokenizer

# ne_tokenizer = NepaliTokenizer()
# en_tokenizer = load_english_tokenizer()

# train_df = pd.read_csv("data/train.csv")

# ne_lengths = train_df['nepali_sent'].apply(lambda x: len(ne_tokenizer.encode(str(x)).ids))
# en_lengths = train_df['english_sent'].apply(lambda x: len(en_tokenizer.encode(str(x)).ids))

# print(f"Nepali  — max: {ne_lengths.max()} | 95th percentile: {ne_lengths.quantile(0.95):.0f} | mean: {ne_lengths.mean():.0f}")
# print(f"English — max: {en_lengths.max()} | 95th percentile: {en_lengths.quantile(0.95):.0f} | mean: {en_lengths.mean():.0f}")


# from src.utils.tokenizer import load_english_tokenizer
# en_tokenizer = load_english_tokenizer()

# # Check how Rana is tokenized
# tokens = en_tokenizer.encode("Rana")
# print("Token IDs:", tokens.ids)
# print("Tokens:", [en_tokenizer.id_to_token(i) for i in tokens.ids])

# Output will be something like:
# Token IDs: [1234, 567]
# Tokens: ['ran', 'Ġa']   ← split into two pieces!
# ```

# **This is exactly why NER + transliteration is important:**
# ```
# Without NER:
# राणाशासनकालमा → model tries to translate → "ran a rule"  ❌

# With NER:
# राणा detected as PERSON/ORG
# → bypasses model completely
# → transliterate directly → "Rana"
# → model only translates: "शासनकालमा" → "during the regime"
# → final: "During the Rana regime"  ✅



# from src.utils.tokenizer import load_english_tokenizer
# en_tokenizer = load_english_tokenizer()

# names = [
#     "Rana", "Prithvi", "Narayan", "Kathmandu", 
#     "Gorkha", "Tenzing", "Hillary", "Sita", "Ram",
#     "Hari", "Krishna", "Buddha", "Lumbini"
# ]

# for name in names:
#     tokens = en_tokenizer.encode(name)
#     pieces = [en_tokenizer.id_to_token(i) for i in tokens.ids]
#     status = "✅" if len(tokens.ids) == 1 else "❌ SPLIT"
#     print(f"{name:<12} → {pieces}  {status}")
# # ```

# **Expected output:**
# ```
# Rana         → ['R', 'ana']           ❌ SPLIT
# Prithvi      → ['Pri', 'th', 'vi']    ❌ SPLIT
# Narayan      → ['Nar', 'ayan']        ❌ SPLIT
# Kathmandu    → ['K', 'ath', 'man', 'du'] ❌ SPLIT
# Hari         → ['H', 'ari']           ❌ SPLIT
# Ram          → ['Ram']                ✅
# ```

# **This confirms — ALL Nepali proper names get split by BPE which causes garbage output.**

# **This is exactly why NER + preprocess is critical for your project:**
# ```
# WITHOUT preprocess:
# राणा → model → BPE splits → ['R','ana'] → "ran a"  ❌
# पृथ्वीनारायण → model → BPE splits → "earth nara yan"  ❌
# काठमाडौं → model → BPE splits → "k ath man du"  ❌

# WITH preprocess:
# राणा → NER detects → transliterate → "Rana" → placeholder __PER0__
#        model never sees the name
#        model only translates normal Nepali words
#        restore placeholder → "Rana"  ✅


# from src.utils.tokenizer import load_nepali_tokenizer, load_english_tokenizer

# ne = load_nepali_tokenizer()
# en = load_english_tokenizer()

# print("── Nepali ──")
# print("PAD:", ne.token_to_id("[PAD]"))  # should be 0
# print("UNK:", ne.token_to_id("[UNK]"))  # should be 1
# print("SOS:", ne.token_to_id("[SOS]"))  # should be 2
# print("EOS:", ne.token_to_id("[EOS]"))  # should be 3

# print("── English ──")
# print("PAD:", en.token_to_id("[PAD]"))  # should be 0
# print("SOS:", en.token_to_id("[SOS]"))  # should be 1
# print("EOS:", en.token_to_id("[EOS]"))  # should be 2
# print("UNK:", en.token_to_id("[UNK]"))  # should be 3

# Check if 'राम' is in your vocabulary
if 'राम' not in vocab:
    print("राम is OOV - model will guess wrong!")
    
# Common Nepali names should be in vocab:
# राम, सीता, कृष्ण, हरि, etc.