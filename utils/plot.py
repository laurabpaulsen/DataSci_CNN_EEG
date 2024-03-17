from matplotlib import pyplot as plt

# set parameters for plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300



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