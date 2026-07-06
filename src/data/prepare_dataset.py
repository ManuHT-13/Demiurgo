"""
Prepares the CIFAR-10 dataset by downloading it through PyTorch and
generating two CSV files: one with the original labels and another with
randomized labels for the different experiments.
The images are stored in .npy format.
"""

import os
import numpy as np
import pandas as pd
import torchvision

SEED = 420
np.random.seed(SEED)

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)


def download_cifar10():
    train = torchvision.datasets.CIFAR10(root=RAW_DIR, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=RAW_DIR, train=False, download=True)
    return train, test


def save_images(train, test):
    np.save(os.path.join(RAW_DIR, 'train_images.npy'), train.data)
    np.save(os.path.join(RAW_DIR, 'test_images.npy'), test.data)
    return train.targets, test.targets


def create_csv(train_labels, test_labels):
    filas = []
    for i, label in enumerate(train_labels):
        filas.append({'split': 'train', 'index': i, 'label': label})
    for i, label in enumerate(test_labels):
        filas.append({'split': 'test', 'index': i, 'label': label})

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RAW_DIR, 'labels.csv'), index=False)
    return df


def create_random_csv(df):
    """
    Creates a CSV with randomly shuffled labels, following the experiment
    described by Zhang et al. (2017),
    "Understanding Deep Learning Requires Rethinking Generalization".
    Only the training split labels are randomized.
    """
    df_random = df.copy()
    mask_train = df_random['split'] == 'train'
    n_train = mask_train.sum()
    df_random.loc[mask_train, 'label'] = np.random.randint(0, 10, size=n_train)
    df_random.to_csv(os.path.join(RAW_DIR, 'labels_random.csv'), index=False)
    return df_random


if __name__ == '__main__':
    train, test = download_cifar10()
    train_labels, test_labels = save_images(train, test)
    df = create_csv(train_labels, test_labels)
    create_random_csv(df)
    print('Dataset ready at', RAW_DIR)