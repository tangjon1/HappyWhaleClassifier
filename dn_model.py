#!/usr/bin/env python

## Importing packages - Please DO NOT alter this box ##
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
torch.manual_seed(0)

from captum.attr import IntegratedGradients
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import os
import imageio

from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # IMPORTANT for sbatch (no GUI)
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_erosion
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

import wandb #comment this out if you are not using weights and biases
import random #comment this out if you are not using weights and biases

import argparse

parser = argparse.ArgumentParser(description="Simple example of argparse usage.")
parser.add_argument("-i", "--id", help="Numeric identifier for this run", type=int)
parser.add_argument("-e", "--epochs", help="Num epochs", type=int)
parser.add_argument("-l", "--learn-rate", help="Learning rate", type=float)
parser.add_argument("-b", "--batch-size", help="Batch size", type=int)
parser.add_argument("-o", "--output-dir", help="Directory for output files and images")
parser.add_argument("-p", "--optimizer", help="Select optimizer", choices=["adam", "sgd"])
parser.add_argument("-r", "--resolution", help="Image resolution to train on", type=int)

args = parser.parse_args()

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = os.path.join(args.output_dir, f"run_{RUN_ID}")
os.makedirs(OUTDIR, exist_ok=True)

def render_args(args: argparse.Namespace) -> str:
    '''Render the run mode (e.g. 'sort', 'consensus') and all set arguments as a string, separated by newlines'''
    arg_summary: str = ""
    for parameter, argument in vars(args).items():
        arg_summary += f"\t{parameter}: {argument}\n"
    return arg_summary

print(render_args(args))

# Set device to cuda if it's available otherwise default to "cpu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

images = "/projects/bgmp/shared/Bi625/ML_Assignment/Datasets/Whale_species/species"

from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # robust to grayscale/RGBA
    transforms.RandomResizedCrop(
        args.resolution, scale=(0.6, 1.0), ratio=(0.75, 1.33),
        interpolation=InterpolationMode.BILINEAR
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
        p=0.6
    ),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    # Works well for robustness; doesn’t require PIL.
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(args.resolution),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

all_images = datasets.ImageFolder(images)

train_size = int(0.7 * len(all_images))
val_size = int(0.15 * len(all_images))
test_size = len(all_images) - (train_size + val_size)
print(train_size, val_size, test_size)
assert train_size + val_size + test_size == len(all_images)

#train_set, val_set, test_set = torch.utils.data.random_split(all_images, [train_size, val_size, test_size])
train_idx, val_idx, test_idx = torch.utils.data.random_split(
    range(len(all_images)),
    [train_size, val_size, test_size]
)
# sapply transform to training data and validation data
train_dataset = datasets.ImageFolder(images, transform=train_transform)
val_dataset   = datasets.ImageFolder(images, transform=val_transform)
test_dataset  = datasets.ImageFolder(images, transform=val_transform)

# subset the data
train_set = torch.utils.data.Subset(train_dataset, train_idx.indices)
val_set   = torch.utils.data.Subset(val_dataset, val_idx.indices)
test_set  = torch.utils.data.Subset(test_dataset, test_idx.indices)

def _get_weights(subset,full_dataset):
    ys = np.array([y for _, y in subset])
    counts = np.bincount(ys)
    # Can use resent pretrained model weight here
    # TODO: 
    label_weights = 1.0 / counts
    weights = label_weights[ys]

    print("Number of images per class:")
    for c, n, w in zip(full_dataset.classes, counts, label_weights):
        print(f"\t{c}:\tn={n}\tweight={w}")
        
    return weights

train_weights = _get_weights(train_set,all_images)
train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

# ----------- Predict and Evaluate--------
# predict the test dataset
def predict(model, dataset):
    dataset_prediction = []
    dataset_groundtruth = []
    model = model
    with torch.no_grad():
        for x, y_true in dataset:
            inp = x[None]
            y_pred = model(inp)
            dataset_prediction.append(y_pred.argmax().cpu().numpy())
            dataset_groundtruth.append(y_true)
    
    return np.array(dataset_prediction), np.array(dataset_groundtruth)
            
    # create seaborn heatmap with required labels
    ax=sns.heatmap(cm, annot=annot, fmt='', vmax=30, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    ax.set_title(title)

# Plot confusion matrix 
# orginally from Runqi Yang; 
# see https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
def cm_analysis(y_true, y_pred, title,
                save_path="confusion_matrix.png",
                outdir=OUTDIR,
                figsize=(10,10),
                log_wandb=False):

    """
    Generate and save confusion matrix plot with annotations.

    Args:
        y_true: true labels (nsamples,)
        y_pred: predicted labels (nsamples,)
        title:  plot title
        save_path: filename for saved plot
        outdir: directory to save plot
        figsize: figure size
        log_wandb: if True, logs image to wandb
    """

    os.makedirs(outdir, exist_ok=True)
    full_path = os.path.join(outdir, save_path)

    labels = ['Beluga','Common dolphin', 'False killer whale',
              'Fin whale', 'Gray whale','Humpback whale']

    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape

    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = f"{p:.1f}%\n{c}/{s[0]}"
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{p:.1f}%\n{c}"

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_df,
        annot=annot,
        fmt='',
        cmap="viridis",
        cbar=True,
        ax=ax
    )

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix saved to: {full_path}")

    if log_wandb:
        import wandb
        wandb.log({"confusion_matrix": wandb.Image(full_path)})

    return full_path

#---------------PARAMETERS-------------------
batchsize = args.batch_size
learning_rate=args.learn_rate
epochs=args.epochs

train_loader = DataLoader(train_set, batch_size=batchsize, drop_last=True, sampler=train_sampler)
val_loader = DataLoader(val_set, batch_size=batchsize, drop_last=True, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batchsize, drop_last=True, shuffle=True)

# MODEL
from torchvision.models import resnet18, ResNet18_Weights

# Load pretrained weights
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

# Replace final layer
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU
model = model.to(device)

# Define loss function & optimizer
criterion = nn.CrossEntropyLoss()

match args.optimizer:
    case 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    case 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# ----------- Run Model ----------------------
wandb.init(
    entity="BGMP_HappyWhale",
    project="BGMP_HappyWhale",
    name=f"j-model-{RUN_ID}", ##update this with your name
    config={
        "learning rate": learning_rate, # possibly update
        "architecture": "CNN",
        "dataset": "Species",
        "epochs": epochs,
        "batch_size": batchsize,
        "optimizer": args.optimizer
    }
) 

batch=0
num_epochs = epochs
train_losses, train_acc_list, val_losses, val_acc_list = [], [], [],[]

for epoch in range(num_epochs):
    # Setting model to "training mode"
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    # For each batch of data within the loader
    for inputs, labels in train_loader:
        # Send our input images and their labels to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Inputting our training images into the model
        # and Predicting the image classification label
        outputs = model(inputs)
        # Figuring out the loss from our predictions
        loss = criterion(outputs, labels)
        # Compute gradients (aka backward pass)
        loss.backward()
        # Update model parameters
        optimizer.step()

        # Adding the loss to our running sum
        # Loss is calculated for each batch within an epoch
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        batch+=1
        print(f'Epoch [{epoch+1}/{num_epochs}] | Batch #{batch} | Batch Accuracy {(correct/total)*100:.2f}%')


    # Getting metrics for our training pass 
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_acc_list.append(train_acc)
    
    # Switching our model to "evaluation mode"
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    # Disable gradient calculation b/c we are evalulating the model
    with torch.no_grad():
        # Load in batches of our validation data
        for inputs, labels in val_loader:
            # Send test images and labels to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            # Predict the image classification label
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            # Figuring out how many predicted labels = true labels
            correct += predicted.eq(labels).sum().item()

            # Figuring out the loss from our predictions
            loss = criterion(outputs, labels)
            # Adding the loss to our running sum
            # Loss is calculated for each batch within an epoch
            running_loss += loss.item() * inputs.size(0)
    # Getting our accuracy from our test data
    val_acc = 100. * correct / total
    val_acc_list.append(val_acc)
    # Getting the loss from our test data
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.2f}%')

    # log metrics to wandb
    wandb.log({"validation_accuracy": val_acc, "validation_loss": val_loss, "train_loss":train_loss, "epoch": epoch + 1})

wandb.finish()

# --- Save accuracy curve ---
plt.figure()
plt.plot(range(1, epochs + 1), val_acc_list, label="Val Acc")
plt.plot(range(1, epochs + 1), train_acc_list, label="Train Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epoch")
plt.legend()
acc_path = os.path.join(OUTDIR, "accuracy_curve.png")
plt.savefig(acc_path, dpi=200, bbox_inches="tight")
plt.close()

# --- Save loss curve ---
plt.figure()
plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
loss_path = os.path.join(OUTDIR, "loss_curve.png")
plt.savefig(loss_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved plots to:\n  {acc_path}\n  {loss_path}")

y_pred, y_true = predict(model, test_set)
cm_analysis(y_true, y_pred, "Confusion matrix")

#----------- DONE--------------
# save model
PATH = f'model_{RUN_ID}.pt'
torch.save(model, PATH)