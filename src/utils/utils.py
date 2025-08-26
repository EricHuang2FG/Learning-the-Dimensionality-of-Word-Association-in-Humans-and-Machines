import string
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    mag_a, mag_b = np.linalg.norm(vector_a), np.linalg.norm(vector_b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return (np.dot(vector_a, vector_b) / (mag_a * mag_b))


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


def parse_association_data(
    filename: str, delim: str = "\t"
) -> dict[str, dict[str, float]]:
    association_data = {}
    with open(filename, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index == 0:
                continue
            line = line.strip().lower().split(delim)

            # assume each line starts with word, association
            # and ends with prob
            word, association, *_, prob = line
            prob = float(prob)

            if word in association_data:
                association_data[word][association] = prob
            else:
                association_data[word] = {association: prob}

    return association_data
