# this is the complete python script to train the model

# ====================================================================================================
# import required libraries
# general libraries
import numpy as np
import os, sys
import torch
import matplotlib.pyplot as plt

# datasets
import torchvision.datasets as datasets
# transforms
import torchvision.transforms as transforms
# neural network modules
import torch.nn as nn
# PyTorch utilities
from torch.utils.data import DataLoader
# optimizers
import torch.optim as optim

# other misc libraries
import importlib
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src import utilities
from src.utilities import (train_model, evaluate,
                           plot_metrics, prediction_and_image,
                           print_conclusion, realworld_prediction)
importlib.reload(utilities)

# ====================================================================================================
# set up device to be used for computation
device = "cuda" if torch.cuda.is_available else "cpu"

# ====================================================================================================
# define image transform to load data
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)]
)

# ====================================================================================================
# load the dataset
train_dataset = datasets.MNIST(root="../data", train=True, transform=image_transform)
test_dataset = datasets.MNIST(root="../data", train=False, transform=image_transform)

# ====================================================================================================
# build a CNN Model by subclassing `nn.Module`
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=24,kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=24,out_channels=48,kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()

        # Add regularization (here, dropout)
        self.dropout = nn.Dropout(p=0.6)

        self.fc_linear_1 = nn.Linear(in_features=48*5*5, out_features=512)
        self.fc_linear_2 = nn.Linear(in_features=512, out_features=256)
        self.fc_linear_3 = nn.Linear(in_features=256, out_features=128)
        self.fc_linear_4 = nn.Linear(in_features=128, out_features=64)
        self.fc_linear_5 = nn.Linear(in_features=64, out_features=10)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(X)
        X = self.pool1(X)
        
        X = self.conv2(X)
        X = self.relu(X)
        X = self.pool2(X)

        X = self.flatten(X)
        X = self.dropout(X)
        
        X = self.fc_linear_1(X)
        X = self.sigmoid(X)
        X = self.fc_linear_2(X)
        X = self.sigmoid(X)
        X = self.fc_linear_3(X)
        X = self.sigmoid(X)
        X = self.fc_linear_4(X)
        X = self.sigmoid(X)
        X = self.fc_linear_5(X)
        
        return X

# ====================================================================================================
# train the model
# instantiate a model and load to `device`
torch.manual_seed(42)
model = CNN().to(device)

# HYPERPARAMETERS
BATCHSIZE = 128
EPOCHS = 20
LR = 0.01

# dataset loaders
trainset_loader = DataLoader(train_dataset, BATCHSIZE, shuffle=True)
testset_loader = DataLoader(test_dataset, BATCHSIZE, shuffle=False)

# cost function and optimizer
cost_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LR)

# start training
print("="*100)
print(f"Starting to train the model...")
losses, accuracies = train_model(model, trainset_loader, cost_func, optimizer, EPOCHS)
print("Training completed...!")
plot_metrics(losses, accuracies,"CNN6")

# ====================================================================================================
# add a check point for future reference
CHECKPOINT = Path("checkpoints")
CHECKPOINT.mkdir(parents=True, exist_ok=True)

CHECKPOINT_NAME = "CNN_06.pt"
CHECKPOINT_PATH = CHECKPOINT/CHECKPOINT_NAME

torch.save(
    obj={
        'epochs' : EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses, # Save the final loss
        'accuracies': accuracies, # Save the final accuracy
        'batch_size': BATCHSIZE,
        'learning_rate': LR
    },
    f=CHECKPOINT_PATH
)

print(f"Checkpoint saved to : {CHECKPOINT_PATH}")