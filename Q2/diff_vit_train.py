import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import numpy as np

import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import os
import yaml
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from img_transforms import get_augmentation_set


with open("vit_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Assign the config values to the corresponding variables
d_model   = config["d_model"]
n_classes = config["n_classes"]
img_size  = config["img_size"]
patch_size = config["patch_size"]
n_channels = config["n_channels"]
n_heads   = config["n_heads"]
n_layers  = config["n_layers"]
batch_size = config["batch_size"]
epochs    = config["epochs"]
alpha     = config["alpha"]
num_layers = config["num_layers"]
pos_type = config["pos_type"]
aug_type = config["aug_type"]

exp_name = f"Dif-vit-patchsize-{patch_size[0]}-attention_head-{n_heads}-layer-{n_layers}-aug-type-{aug_type}-pos_tyep-{pos_type}"

wandb.init(project = "vit-image-classification", name = exp_name)

config = {
    "d_model": d_model,
    "n_classes": n_classes,
    "img_size": img_size,
    "patch_size": patch_size,
    "n_channels": n_channels,
    "n_heads": n_heads,
    "n_layers": n_layers,
    "batch_size": batch_size,
    "epochs": epochs,
    "alpha": alpha
}

wandb.config.update(config)


# transform = T.Compose([
#   # T.Resize(img_size),
#   T.ToTensor()
# ])

transform = get_augmentation_set(aug_type)

train_set = CIFAR10(
  root="/home/akash/ws/cv_assignment/assignment-5-MlLearnerAkash/Q1/dataset", train=True, download=True, transform=transform
)
test_set = CIFAR10(
  root="/home/akash/ws/cv_assignment/assignment-5-MlLearnerAkash/Q1/dataset", train=False, download=True, transform=transform
)


train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)


from diff_vit import VisionTransformer


def train_transformer(transformer,save_path, criterion, epochs, optimizer):
   
    # Setup
    init_val_loss = np.inf
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, 
          f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    
    # Ensure the model is on the proper device.
    transformer.to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    # Training & Validation loop
    for epoch in range(epochs):
        transformer.train()
        training_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc = f'Epoch {epoch+1}/ {epochs} - Training', leave = False)
        # Training loop
        j =0
        for i, (inputs, labels) in enumerate(train_loader_tqdm, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs,_ = transformer(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())
            j+=1
            

            


        scheduler.step()
        
        avg_loss = training_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs} - Train loss: {avg_loss:.3f}')
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
        
        # Validation loop
        transformer.eval()
        validation_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            for val_inputs, val_labels in test_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs,_ = transformer(val_inputs)
                _,predicted = torch.max(val_outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

                val_loss = criterion(val_outputs, val_labels)
                validation_loss += val_loss.item()
        
        avg_val_loss = validation_loss / len(test_loader)
        print(f'Epoch {epoch + 1}/{epochs} - Validation loss: {avg_val_loss:.3f}')
        wandb.log({"validation_loss": avg_val_loss})

        val_accuracy = correct/total
        wandb.log({"validation_accuracy": val_accuracy})
        
        # Save best model based on validation loss
        if avg_val_loss < init_val_loss:
            init_val_loss = avg_val_loss
            torch.save(transformer.state_dict(), os.path.join(save_path, "best.pt"))
        
        # Log a few sample predictions from the last validation batch.
        sample_inputs = val_inputs[:4].detach().cpu()
        sample_labels = val_labels[:4].detach().cpu()
        sample_outputs = val_outputs[:4].detach().cpu()
        _, sample_preds = torch.max(sample_outputs, 1)
        
        samples = []
        for idx in range(len(sample_inputs)):
            # Convert image from (C, H, W) to (H, W, C) for plotting.
            image_np = sample_inputs[idx].permute(1, 2, 0).numpy()
            plt.figure(figsize=(2,2))
            plt.imshow(image_np)
            plt.title(f"GT: {sample_labels[idx].item()} | Pred: {sample_preds[idx].item()}")
            plt.axis("off")
            fig = plt.gcf()
            samples.append(wandb.Image(fig))
            plt.close(fig)
        
        wandb.log({"sample_predictions": samples, "epoch": epoch + 1})

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transformer = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers,num_layers, pos_type).to(device)

    save_path = exp_name
    criterion = nn.CrossEntropyLoss()
    epochs = epochs
    optimizer = Adam(transformer.parameters(), lr=alpha)
    train_transformer(transformer = transformer,
                        save_path=save_path, 
                        criterion=criterion, 
                        epochs=epochs, 
                        optimizer=optimizer)

    transformer.eval()  # Set to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs,_ = transformer(images)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Accumulate predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        print(f'\nModel Accuracy: {accuracy:.2f} %')
        wandb.log({"test_accuracy": accuracy})

    # Compute confusion matrix using sklearn
    cm = confusion_matrix(all_labels, all_preds)


    # Plot confusion matrix as a heatmap for logging.
    def plot_confusion_matrix(cm, classes,
                            normalize=False,

                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        return plt.gcf()

    # Define class names for the confusion matrix (modify as needed)
    class_names = [str(i) for i in range(n_classes)]
    cm_fig = plot_confusion_matrix(cm, classes=class_names, title="Confusion Matrix")

    # Log the confusion matrix figure with wandb
    wandb.log({"confusion_matrix": wandb.Image(cm_fig)})

    # Optionally, you can finish the wandb run:
    wandb.finish()
