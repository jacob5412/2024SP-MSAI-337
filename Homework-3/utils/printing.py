from prettytable import PrettyTable


def print_rouge_scores(scores):
    table = PrettyTable()
    table.field_names = ["Metric", "Type", "Precision", "Recall", "F-measure"]

    for key, value in scores.items():
        for score_type, score in zip(
            ["low", "mid", "high"], [value[0], value[1], value[2]]
        ):
            row = [
                key,
                score_type,
                f"{score[0]:.4f}",
                f"{score[1]:.4f}",
                f"{score[2]:.4f}",
            ]
            table.add_row(row)

    print(table)
