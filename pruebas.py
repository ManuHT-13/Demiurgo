# borrar para entregar
# archivo para hacer lineas de codigo extras

import pandas as pd

df = pd.read_csv("data/raw/labels.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(3))
print(df.tail(3))