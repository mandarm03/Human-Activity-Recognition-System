import numpy as np
import pandas as pd

# Path to train inertial signals
PATH = "data\\UCI HAR Dataset\\train\\Inertial Signals\\"

signals = [
    "body_acc_x_train.txt",
    "body_acc_y_train.txt",
    "body_acc_z_train.txt",
    "body_gyro_x_train.txt",
    "body_gyro_y_train.txt",
    "body_gyro_z_train.txt",
    "total_acc_x_train.txt",
    "total_acc_y_train.txt",
    "total_acc_z_train.txt",
]

data = []

for signal in signals:
    df = pd.read_csv(PATH + signal, sep=r"\s+", header=None)

    print(f"{signal}: {df.shape}")

    data.append(df.values)

# Convert list to numpy array
X_train = np.transpose(np.array(data), (1, 2, 0))

print("\nFinal Shape:", X_train.shape)
