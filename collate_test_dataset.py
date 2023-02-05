import pandas as pd
import torch
from torch.utils.data import Dataset


class Alphabet_Dataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()

        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, a):
        return self.data.iloc[a]


def custom_collate_fn(batch):

    # Define a tensor of the same size as our image batch to store loaded images into

    image_tensors = []
    labels = []

    for item in batch:
        # load a single image
        image_tensor = torch.load(f"{item[0]}").unsqueeze(0)
        # put image into a list
        image_tensors.append(image_tensor)
        # put the same image's label into another list
        labels.append(item[1])

    image_batch_tensor = torch.stack(image_tensors)
    # Use the label list to create a torch tensor of ints
    label_batch_tensor = torch.LongTensor(labels)
    return (image_batch_tensor, label_batch_tensor)
