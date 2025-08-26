from src.utils.constants import DEFAULT_EMBEDDING_DIMENSION
from gensim.models import Word2Vec, KeyedVectors


def load_word_to_vec_model(model_directory: str) -> KeyedVectors:
    return KeyedVectors.load(model_directory)


def train_word_to_vec_model(
    corpus: str,
    dest: str,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    save: bool = True,
) -> KeyedVectors | None:
    all_words = []
    with open(corpus, "r", encoding="utf-8") as r:
        for index, line in enumerate(r):
            if index == 0:
                continue

            all_words.append(line.strip().split(","))

    model = Word2Vec(
        sentences=all_words,
        vector_size=dimension,
        sg=1,
    )
    if not save:
        return model.wv
    model.wv.save(dest)


if __name__ == "__main__":
    # train_word_to_vec_model(
    #     "data/SWOW/SWOW_flattened.R100.csv", "src/models/SWOE/SWOE.v1.model"
    # )
    model = load_word_to_vec_model("src/models/SWOE/SWOE.v1.model")
    print(model.most_similar("hello"))
