import matplotlib.pyplot as plt
import os


def plot_loss(train_losses, val_losses, save_dir):
    """
    Plot training and validation loss curves and save to file.

    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_dir (str): Directory to save the plot image
    """
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def plot_acc(train_accs, val_accs, save_dir):
    """
    Plot training and validation accuracy curves and save to file.

    Args:
        train_accs (list): List of training accuracies per epoch
        val_accs (list): List of validation accuracies per epoch
        save_dir (str): Directory to save the plot image
    """
    plt.figure()
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'accuracy_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy curve saved to {save_path}")