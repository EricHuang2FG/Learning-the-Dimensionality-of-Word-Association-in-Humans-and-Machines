import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from gensim.models import KeyedVectors
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


def p_value_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"

    return ""


def get_perason_correlation(
    association_probability_approx_type: str,
    association_strength_data_path: str = "data/SWOW/strength.SWOW-EN.R123.csv",
    corpus: str = "data/SWOW/SWOW_flattened.R100.csv",
    dim: int = DEFAULT_EMBEDDING_DIMENSION,
) -> tuple[float, float]:
    if association_probability_approx_type not in [
        ASSOCIATION_PROBABILITY_APPROX_TYPE_COSINE_SIMILARITY,
        ASSOCIATION_PROBABILITY_APPROX_TYPE_SOFTMAX,
    ]:
        raise ValueError(
            f"Invalid association_probability_approx_type: {association_probability_approx_type}"
        )

    word_associations: dict[str, dict[str, float]] = parse_association_data(
        association_strength_data_path
    )
    word_to_vec_model: KeyedVectors = train_word_to_vec_model(
        corpus, None, dimension=dim, save=False
    )
    print("A word2vec model training was completed.")

    approximated_association_probabilities: list[float] = []
    actual_association_probabilities: list[float] = []

    cache: dict[str, float] = {}
    for cue, responses in word_associations.items():
        for response, probability_of_association in responses.items():
            # preserves relative order between the two lists
            # i.e. each value in the list coresponds to the same cue and response
            approximated_association_probabilities.append(
                get_cosine_similarity_association_probability(
                    cue, response, word_to_vec_model
                )
                if association_probability_approx_type
                == ASSOCIATION_PROBABILITY_APPROX_TYPE_COSINE_SIMILARITY
                else get_softmax_association_probability(
                    cue, response, responses, word_to_vec_model, cache
                )
            )
            actual_association_probabilities.append(probability_of_association)

    return pearsonr(
        approximated_association_probabilities, actual_association_probabilities
    )


def plot_graph(
    graph_destination: str = "results/graphs/reconstruction_correlation_vs_dimension.pdf",
    association_strength_data_path: str = "data/SWOW/strength.SWOW-EN.R123.csv",
    corpus: str = "data/SWOW/SWOW_flattened.R100.csv",
) -> None:
    FIGURE_SIZE: tuple[int, int] = (12, 18)
    Y_LIM: tuple[float, float] = (0.0, 0.5)

    x: list[int] = [10 * (2**i) for i in range(7)]
    # 10, 20, 40, 80, 160, 320, 640

    y_cosine_similarity: list[float] = []
    p_cosine_similarity: list[float] = []

    y_softmax: list[float] = []
    p_softmax: list[float] = []

    for curr_dim in x:
        curr_corr_cosine_similarity, curr_p_cosine_similarity = get_perason_correlation(
            ASSOCIATION_PROBABILITY_APPROX_TYPE_COSINE_SIMILARITY,
            association_strength_data_path=association_strength_data_path,
            corpus=corpus,
            dim=curr_dim,
        )
        curr_corr_softmax, curr_p_softmax = get_perason_correlation(
            ASSOCIATION_PROBABILITY_APPROX_TYPE_SOFTMAX,
            association_strength_data_path=association_strength_data_path,
            corpus=corpus,
            dim=curr_dim,
        )

        y_cosine_similarity.append(curr_corr_cosine_similarity)
        p_cosine_similarity.append(curr_p_cosine_similarity)

        y_softmax.append(curr_corr_softmax)
        p_softmax.append(curr_p_softmax)

    # plot graphs
    _, (ax_1, ax_2) = plt.subplots(2, 1, figsize=FIGURE_SIZE)

    # plot graph with cosine similarity approximation
    ax_1.bar(x, y_cosine_similarity, color="blue", width=1.0)
    ax_1.set_title("Cosine Similarity", fontweight="bold", fontsize=17)

    for index, (corr, p) in enumerate(zip(y_cosine_similarity, p_cosine_similarity)):
        ax_1.text(x[index], corr + 0.01, p_value_to_stars(p), ha="center", fontsize=10)

    # plot graph with softmax approximation
    ax_2.bar(x, y_softmax, color="blue", width=1.0)
    ax_2.set_title("Softmax", fontweight="bold", fontsize=17)

    for index, (corr, p) in enumerate(zip(y_softmax, p_softmax)):
        ax_2.text(x[index], corr + 0.01, p_value_to_stars(p), ha="center", fontsize=10)

    # general settings
    for ax in (ax_1, ax_2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Embeddings Dimension", fontweight="bold", fontsize=14)
        ax.set_ylabel("Correlation of Reconstruction", fontweight="bold", fontsize=14)
        ax.set_ylim(*Y_LIM)
        ax.grid(True)

    plt.savefig(graph_destination, bbox_inches="tight", pad_inches=0.5)
    plt.show()


if __name__ == "__main__":
    plot_graph(
        graph_destination="results/graphs/reconstruction_correlation_vs_dimension.pdf"
    )
