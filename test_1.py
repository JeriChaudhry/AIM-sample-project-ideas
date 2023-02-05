import torch
import torch.nn.functional as F
import torch.nn as nn

import torchvision

import unittest
from PIL import Image
from torch.utils.data import DataLoader
from collate_test_dataset import *
from util import get_transform, get_acc
from Functions.CNN import *


IMG_PATH = "test_image.jpg"


class TestProject(unittest.TestCase):
    def test_transform(self):
        img = Image.open(IMG_PATH)
        transform = get_transform()
        transformed_img = transform(img)
        self.assertEqual(transformed_img.shape[0], 3)
        self.assertEqual(transformed_img.shape[1], 28)
        self.assertEqual(transformed_img.shape[2], 28)

    def test_acc(self):
        test_dataset = Alphabet_Dataset("Test_Dataset.csv")

        model_path = r"Models_Checkpoint/Checkpoint_10.pth"

        # Load dataset

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=32,
            collate_fn=custom_collate_fn,
            shuffle=True,
        )
        loss_fn = nn.CrossEntropyLoss()

        # Load the Model
        model = Recognising_Letters(input_shape=1, hidden_units=12, output_shape=26)
        model = torch.load(f=model_path)

        # Get the accuracy
        acc = get_acc(model, test_loader, loss_fn)
        self.assertGreater(acc, 0.7)

       