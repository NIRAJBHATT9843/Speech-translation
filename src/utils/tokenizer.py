# src/utils/tokenizer.py

from tokenizers import Tokenizer
from src.utils.nepali_tokenizer import NepaliTokenizer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_nepali_tokenizer():
    # Use your custom syllable-based tokenizer instead of JSON file
    return NepaliTokenizer()

def load_english_tokenizer():
    path = os.path.join(MODEL_DIR, "en_tokenizer.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"English tokenizer not found: {path}")
    return Tokenizer.from_file(path)
