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
    """
    A class for evaluating RAG-based models using various metrics and generating evaluation plots.

    Methods:
    - evaluate_ragas(dataset): Evaluates the RAG-based model on the dataset using specified metrics.
    - plot_evaluation(result, filepath): Plots the evaluation results as a heatmap and saves it to a file.

    Args:
    - dataset: The dataset to evaluate the model on.
    - result: The evaluation result to plot.
    - filepath (str): The file path to save the evaluation plot.
    """

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
