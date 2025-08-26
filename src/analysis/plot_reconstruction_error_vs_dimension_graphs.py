import matplotlib.pyplot as plt
from src.utils.constants import (
    DEFAULT_EMBEDDING_DIMENSION,
    ASSOCIATION_PROBABILITY_APPROX_TYPE_COSINE_SIMILARITY,
    ASSOCIATION_PROBABILITY_APPROX_TYPE_SOFTMAX,
)
from src.utils.utils import (
    parse_association_data,
    get_cosine_similarity_association_probability,
    get_softmax_association_probability,
)
from src.models.word_to_vec_model import train_word_to_vec_model
from gensim.models import KeyedVectors


def get_reconstruction_error(
    association_probability_approx_type: str,
    association_strength_data_path: str = "data/SWOW/strength.SWOW-EN.R123.csv",
    corpus: str = "data/SWOW/SWOW_flattened.R100.csv",
    dim: int = DEFAULT_EMBEDDING_DIMENSION,
) -> float:
    if association_probability_approx_type not in [
        ASSOCIATION_PROBABILITY_APPROX_TYPE_COSINE_SIMILARITY,
        ASSOCIATION_PROBABILITY_APPROX_TYPE_SOFTMAX,
    ]:
        raise ValueError(
            f"Invalid association_probability_approx_type: {association_probability_approx_type}"
        )

    total_squared_error: float = 0.0
    word_associations: dict[str, dict[str, float]] = parse_association_data(
        association_strength_data_path
    )
    word_to_vec_model: KeyedVectors = train_word_to_vec_model(
        corpus, None, dimension=dim, save=False
    )
    print("A word2vec model training was completed.")

    total_count: int = 0
    cache: dict[str, float] = {}
    for cue, responses in word_associations.items():
        for response, probability_of_association in responses.items():
            total_count += 1
            association_probability_approximation = (
                get_cosine_similarity_association_probability(
                    cue, response, word_to_vec_model
                )
                if association_probability_approx_type
                == ASSOCIATION_PROBABILITY_APPROX_TYPE_COSINE_SIMILARITY
                else get_softmax_association_probability(
                    cue, response, responses, word_to_vec_model, cache
                )
            )

            total_squared_error += (
                association_probability_approximation - probability_of_association
            ) ** 2

    return total_squared_error / total_count


def plot_graph(
    graph_destination: str = "results/graphs/reconstruction_error_vs_dimension.pdf",
    association_strength_data_path: str = "data/SWOW/strength.SWOW-EN.R123.csv",
    corpus: str = "data/SWOW/SWOW_flattened.R100.csv",
) -> None:
    FIGURE_SIZE: tuple[int, int] = (12, 18)
    LINE_WIDTH: int = 3

    x: list[int] = [10 * (2**i) for i in range(7)]
    # 10, 20, 40, 80, 160, 320, 640

    y_cosine_similarity: list[float] = [
        get_reconstruction_error(
            ASSOCIATION_PROBABILITY_APPROX_TYPE_COSINE_SIMILARITY,
            association_strength_data_path=association_strength_data_path,
            corpus=corpus,
            dim=curr_dim,
        )
        for curr_dim in x
    ]

    y_softmax: list[float] = [
        get_reconstruction_error(
            ASSOCIATION_PROBABILITY_APPROX_TYPE_SOFTMAX,
            association_strength_data_path=association_strength_data_path,
            corpus=corpus,
            dim=curr_dim,
        )
        for curr_dim in x
    ]

    # plot graphs
    _, (ax_1, ax_2) = plt.subplots(2, 1, figsize=FIGURE_SIZE)

    # plot graph with cosine similarity approximation
    ax_1.plot(
        x, y_cosine_similarity, color="purple", linewidth=LINE_WIDTH, linestyle=":"
    )
    ax_1.set_title("Cosine Similarity", fontweight="bold", fontsize=17)

    # plot graph with softmax approximation
    ax_2.plot(x, y_softmax, color="purple", linewidth=LINE_WIDTH, linestyle=":")
    ax_2.set_title("Softmax", fontweight="bold", fontsize=17)

    # general settings
    for ax in (ax_1, ax_2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Embeddings Dimension", fontweight="bold", fontsize=14)
        ax.set_ylabel(
            "Mean-Squared Error of Reconstruction", fontweight="bold", fontsize=14
        )
        ax.grid(True)

    plt.savefig(graph_destination, bbox_inches="tight", pad_inches=0.5)
    plt.show()


if __name__ == "__main__":
    plot_graph(graph_destination="results/graphs/reconstruction_error_vs_dimension.pdf")
