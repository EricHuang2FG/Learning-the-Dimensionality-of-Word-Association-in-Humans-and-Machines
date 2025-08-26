import string
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import KeyedVectors


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    mag_a, mag_b = np.linalg.norm(vector_a), np.linalg.norm(vector_b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return np.dot(vector_a, vector_b) / (mag_a * mag_b)


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


def get_cosine_similarity_association_probability(
    w_1: str, w_2: str, word_to_vec_model: KeyedVectors
) -> float:
    sim: float = 0.0
    if w_1 in word_to_vec_model and w_2 in word_to_vec_model:
        sim = word_to_vec_model.similarity(w_1, w_2)

    return sim


def get_softmax_association_probability(
    cue: str,
    response: str,
    all_responses: dict[str, float],
    word_to_vec_model: KeyedVectors,
    denominator_cache: dict[str, float],
) -> float:
    # calculate P(response | cue)
    if cue not in word_to_vec_model or response not in word_to_vec_model:
        # numerator guaranteed to be zero
        # hence just return
        return 0.0

    numerator: float = np.exp(
        np.dot(word_to_vec_model[cue], word_to_vec_model[response])
    )

    if cue in denominator_cache:
        return numerator / denominator_cache[cue]

    denominator: float = 0.0
    for curr_response in all_responses.keys():
        if curr_response in word_to_vec_model:
            denominator += np.exp(
                np.dot(word_to_vec_model[curr_response], word_to_vec_model[cue])
            )

    denominator_cache[cue] = denominator

    return numerator / denominator
