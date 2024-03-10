from matplotlib import pyplot as plt


def plot_history(history, save_path = None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(history['train_loss'], label="Train Loss")
    ax[0].plot(history['val_loss'], label="Validation Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history['train_acc'], label="Train Accuracy")
    ax[1].plot(history['val_acc'], label="Validation Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)