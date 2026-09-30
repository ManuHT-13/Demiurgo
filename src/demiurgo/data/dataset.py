"""
This file defines how PyTorch will load the CIFAR-10 images and their
correct labels in order to train the first neural network

A PyTorch "Dataset" is a class that tells PyTorch how many samples are
available (__len__) and how to retrieve the i-th sample (__getitem__)

PyTorch then uses a "DataLoader", which repeatedly calls __getitem__,
groups the samples into batches, and feeds them to the neural network
"""

from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
 

class CIFAR10SupervisedDataset(Dataset):
    """
    Dataset for supervised learning using the correct labels
    """

    def __init__(self, images_path, labels_path, transform=None):
        images_path = Path(images_path)
        labels_path = Path(labels_path)

        # Load the NumPy array exactly as it was saved to disk
        self.images = np.load(images_path)

        #  Read the labels CSV file using pandas (a table similar to Excel)
        df = pd.read_csv(labels_path)

        # CASE 1: The CSV includes a column indicating whether each row belongs to the
        # training or test set
        split_col = None
        for candidate in ("split", "set", "subset"):
            if candidate in df.columns:
                split_col = candidate
                break
 
        if split_col is not None:
            wanted_split = "train" if "train" in images_path.stem else "test"
            df = df[df[split_col].astype(str).str.lower() == wanted_split].reset_index(drop=True)
 
        if "label" in df.columns:
            self.labels = df["label"].to_numpy()
        else:
            self.labels = df.select_dtypes(include="number").iloc[:, -1].to_numpy()

        # CASE 2: There was no split column, but the CSV has more rows
        # than images: first the 50,000 training rows, then the 10,000 test rows
        # If we are loading train_images.npy, we keep
        # the first N rows; if it is test_images.npy, we keep the last N
        if len(self.labels) != len(self.images) and len(self.labels) > len(self.images):
            n = len(self.images)
            if "train" in images_path.stem:
                print(
                    f"[dataset.py] WARNING: labels.csv has {len(self.labels)} rows "
                    f"and the images {n}. Asuming the first {n} csv rows "
                    f"belong to the train split."
                )
                self.labels = self.labels[:n]
            else:
                print(
                    f"[dataset.py] WARNING: labels.csv has {len(self.labels)} rows "
                    f"and the images {n}. Asuming the first {n} csv rows "
                    f"belong to the test split."
                )
                self.labels = self.labels[-n:]
 
        assert len(self.images) == len(self.labels), (
            f"Descuadre: {len(self.images)} imágenes vs "
            f"{len(self.labels)} etiquetas. Revisa la salida de "
            f"prepare_dataset.py, algo no encaja. Columnas del csv: "
            f"{df.columns.tolist()}"
        )
 
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (32, 32, 3), uint8
        label = int(self.labels[idx])  # Integer in the range 0-9

        if self.transform:
            image = self.transform(image)
        else:
            # Minimal conversion when no transform is provided:
            # NumPy (H, W, C) uint8 -> Tensor (C, H, W) float in the range [0, 1]
            image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0

        return image, label