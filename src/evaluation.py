import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision,
)


class Evaluator:
    def __init__(self) -> None:
        pass

    def evaluate_ragas(self, dataset):
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_relevancy,
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            raise_exceptions=False,
        )
        result.to_pandas()
        return result

    def plot_evaluation(self, result, filepath):
        df = result.to_pandas()
        heatmap_data = df[
            [
                "context_relevancy",
                "context_precision",
                "context_recall",
                "faithfulness",
                "answer_relevancy",
            ]
        ]
        cmap = LinearSegmentedColormap.from_list("green_red", ["red", "green"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=0.5, cmap=cmap)
        plt.yticks(ticks=range(len(df["question"])), labels=df["question"], rotation=0)
        plt.savefig(filepath)
        plt.show()
