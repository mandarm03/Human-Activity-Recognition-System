import pandas as pd

x_train = pd.read_csv(
    r"data/UCI HAR Dataset/train/X_train.txt", sep=r"\s+", header=None
)

print(x_train.shape)
print(x_train.head())
