from src.utils.constants import DEFAULT_EMBEDDING_DIMENSION
from gensim.models import Word2Vec, KeyedVectors


class WordToVec(
    Word2Vec
):  # a class to wrap around the KeyedVectors object to be compatible with the .wv attribute used in the rest of the implementation

    def __init__(self, keyed_vectors: KeyedVectors) -> None:
        self.wv: KeyedVectors = keyed_vectors


def create_word_to_vec_model(model_directory: str) -> WordToVec:
    return WordToVec(KeyedVectors.load(model_directory))


def train_word_to_vec_model(
    corpus: str, dest: str, dimension: int = DEFAULT_EMBEDDING_DIMENSION
) -> None:
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
    model.wv.save(dest)


if __name__ == "__main__":
    # train_word_to_vec_model(
    #     "data/SWOW/SWOW_flattened.R100.csv", "src/models/SWOE/SWOE.v1.model"
    # )
    model = create_word_to_vec_model("src/models/SWOE/SWOE.v1.model")
    print(model.wv.most_similar("hello"))
