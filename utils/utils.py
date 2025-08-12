import string
from nltk.tokenize import sent_tokenize, word_tokenize


def clean_string(s: str, lower: bool = True) -> str:
    stripped = s.strip().strip("'\".,")
    return stripped.lower() if lower else stripped


def read_lines_from_files(filenames: list) -> str:
    s = ""
    num_files = len(filenames)
    for index, file in enumerate(filenames):
        with open(file, "r", encoding="utf-8") as f:
            s += f"{f.read()}{'. ' if index != num_files - 1 else ' '}"
    return s


def tokenize_string(input_string: str) -> list:
    sentences = sent_tokenize(input_string)
    tokenized_words = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [
            clean_string(w)
            for w in words
            if w not in string.punctuation + "’“”"
            and not w.startswith("'")
            and not w.startswith('"')
            and not w.startswith("“")
        ]
        tokenized_words.append(words)

    return tokenized_words
