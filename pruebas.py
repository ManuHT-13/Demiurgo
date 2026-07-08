# borrar para entregar
# archivo para hacer lineas de codigo extras

import numpy as np, pandas as pd

imgs = np.load("data/raw/train_images.npy")
print("Shape:", imgs.shape, "| dtype:", imgs.dtype, "| min/max:", imgs.min(), imgs.max())

df = pd.read_csv("data/raw/labels.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head())