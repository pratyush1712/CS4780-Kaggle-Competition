import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

# -----------------------------------------------------------
print("loading data...")
training_data = pd.read_csv("./train_data.csv")
testing_data = pd.read_csv("./test_data_s.csv")

# -----------------------------------------------------------
print("processing data...")
X_train = training_data.drop("Group (0-Normal Control, 1 Affected)", axis=1)
X_train = X_train.drop("Index ", axis=1)
y_train = training_data["Group (0-Normal Control, 1 Affected)"]
indices = testing_data["Index "]
X_test = testing_data.drop("Index ", axis=1)
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(X_test)
X_test = imp.transform(X_test)

# -----------------------------------------------------------
model = "linear"
print(f"training {model} model...")
svclassifier = SVC(kernel=f"{model}")
svclassifier.fit(X_train.values, y_train.values)

# -----------------------------------------------------------
print("making predictions...")
y_pred = svclassifier.predict(X_test)


def create_submission(preds, index):
    with open("predictions.csv", "w+") as file:
        for i in range(50):
            file.write(f"{indices[i]} {y_pred[i]}")
