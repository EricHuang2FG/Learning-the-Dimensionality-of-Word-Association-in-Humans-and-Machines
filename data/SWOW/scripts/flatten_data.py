from src.utils.utils import clean_string


def flatten_data(src: str, dest: str) -> None:
    # src and dest must be csv files
    # will throw error if format does not match
    with open(dest, "w", encoding="utf-8") as f:
        f.write("Cue,R1,R2,R3\n")
        with open(src, "r", encoding="utf-8") as r:
            for index, line in enumerate(r):
                # skip csv header
                if index == 0:
                    continue

                # last four elements are wanted
                *_, cue, r_1, r_2, r_3 = line.split(",")
                data = [
                    clean_string(concept, lower=False)
                    for concept in [cue, r_1, r_2, r_3]
                ]
                f.write(f"{','.join(data)}\n")
                # each line: cue,r_1,r_2,r_3


if __name__ == "__main__":
    flatten_data("data/SWOW/SWOW-EN.R100.csv", "data/SWOW/SWOW_flattened.R100.csv")
