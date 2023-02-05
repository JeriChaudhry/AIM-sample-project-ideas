import numpy as np
import struct
import matplotlib.pyplot as plt
import pylab
import os
import pandas as pd
import torchvision.transforms as transforms
import torch
import multiprocessing as mp

import torch.nn as nn

from PIL import Image
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from timeit import default_timer as timer

#device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a convolutional neural network
class Recognising_Letters(nn.Module):
    """
    Model architecture that replicates the TinyVGG
    model from CNN explainer website.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            # Create a conv layer - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # values we can set ourselves in our NN's are called hyperparameters
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # values we can set ourselves in our NN's are called hyperparameters
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units
                * 7
                * 7,  # there's a trick to calculating this...
                out_features=output_shape,
            ),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"Output shape of conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Output shape of conv_block_2: {x.shape}")
        x = self.classifier(x)
        # print(f"Output shape of classifier: {x.shape}")
        return x


def train_step(
    epoch,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    Learning_Rate_Scheduler,
    Summary_Wr,
    device: torch.device,
):
    """Performs a training with model trying to learn on data_loader."""
    train_loss, train_acc = 0, 0
    acc_fn = Accuracy(num_classes=26).to(device)
    # Put model into training mode
    model.train()

    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):

        # Put data on target device
        X, y = X.to(device), y.to(device)
        #print(X.shape)

        # 1. Forward pass (outputs the raw logits from the model)
        y_pred = model(X.to(torch.float32))

        # 2. Calculate loss and accuracy (per batch)
        # TODO
        loss = loss_fn(y_pred, y)
        train_loss += loss  # accumulate loss
        train_acc += acc_fn(y_pred, y)  # accumulate accuracy

        # train_acc += Accuracy(y_pred.argmax(dim=1), y) # go from logits -> prediction labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step (update the model's parameters once *per batch*)
        optimizer.step()

    # Divide total train loss and acc by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}% | LR: {Learning_Rate_Scheduler.get_last_lr ()[0]:.5f}")
    
    Summary_Wr.add_scalar(f"Loss/train", train_loss, epoch)
    Summary_Wr.add_scalar(f"Acc/train", train_acc, epoch)
          
    Learning_Rate_Scheduler.step()


def val_step(
    epoch,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    Summary_Wr,
    Early_Stopper,
    # accuracy_fn,
    device: torch.device,
):
    """Performs a validation loop step on model going over data_loader."""
    Val_loss, Val_acc = 0, 0
    acc_fn = Accuracy(num_classes=26).to(device)
    # Put the model in eval mode
    model.eval()

    # Turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send the data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass (outputs raw logits)
            Val_pred = model(X.to(torch.float32))

            # 2. Calculuate the loss/acc
            Val_loss += loss_fn(Val_pred, y)
            Val_acc += acc_fn(Val_pred, y)
            # test_acc += accuracy_fn(y_true=y,
            #                         y_pred=test_pred.argmax(dim=1)) # go from logits -> prediction labels

        # Adjust metrics and print out
        Val_loss /= len(data_loader)
        Val_acc /= len(data_loader)
        print(f"Val loss: {Val_loss:.5f} | Val acc: {Val_acc:.2f}%")
          
          
        Summary_Wr.add_scalar(f"Loss/Val", Val_loss, epoch)
        Summary_Wr.add_scalar(f"Acc/Val", Val_acc, epoch)
          
          
        
          
        if Early_Stopper.should_stop(Val_acc):
            print (f"\nValidation Accuracy hasn't improved for {Early_Stopper.epoch_counter} epoch, aborting...")
            return 0
        else:
            if Early_Stopper.epoch_counter > 0:
                print (f"Epochs without improvement: {Early_Stopper.epoch_counter}\n")
            return 1
#     return Val_loss

        
        

def T_step(
    epoch,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
):
    """Performs a testing loop step on model going over data_loader."""
    Test_loss, Test_acc = 0, 0
    acc_fn = Accuracy(num_classes=26).to(device)
    # Put the model in eval mode
    model.eval()

    # Turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send the data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass (outputs raw logits)
            Test_pred = model(X.to(torch.float32))

            # 2. Calculuate the loss/acc
            Test_loss += loss_fn(Test_pred, y)
            Test_acc += acc_fn(Test_pred, y)
            # test_acc += accuracy_fn(y_true=y,
            #                         y_pred=test_pred.argmax(dim=1)) # go from logits -> prediction labels

        # Adjust metrics and print out
        Test_loss /= len(data_loader)
        Test_acc /= len(data_loader)
        print(f"Test loss: {Test_loss:.5f} | Test acc: {Test_acc:.2f}%")
