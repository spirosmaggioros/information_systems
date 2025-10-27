import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_training_results(
    train_res: list,
    test_res: list,
    title: str = "Training Process",
    xlabel: str = "Epochs",
    ylabel: str = "Score",
) -> None:
    epochs = range(1, len(train_res) + 1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    plt.plot(epochs, train_res, "o-", color="red", label="Train")
    plt.plot(epochs, test_res, "o-", color="orange", label="Test")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc="best")
    plt.show()
