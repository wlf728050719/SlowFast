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

def plot_map(val_maps, save_dir):
    """
    Plot mAP (Mean Average Precision) curve across epochs and save to file.

    Args:
        val_maps (list): List of mAP values per epoch
        save_dir (str): Directory to save the plot image
    """
    plt.figure()
    plt.plot(val_maps, label='Validation mAP', color='purple')
    plt.title('Validation mAP Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()

    # Save the plot
    save_path = os.path.join(save_dir, 'map_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"mAP curve saved to {save_path}")

def plot_class_ap(ap_list, class_names, save_dir):
    """
    Plot AP (Average Precision) for each class as a bar chart (for the best model).

    Args:
        ap_list (list): AP values for each class
        class_names (list): List of class names
        save_dir (str): Directory to save the plot image
    """
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), ap_list, color='skyblue')
    plt.title('Average Precision (AP) per Class (Best Model)')
    plt.xlabel('Class')
    plt.ylabel('AP')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_dir, 'class_ap.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Class AP bar chart saved to {save_path}")