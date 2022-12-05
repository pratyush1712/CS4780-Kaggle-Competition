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
print("data loaded.\n")

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
print("data processed.\n")

# -----------------------------------------------------------
model = "linear"
print(f"training {model} model...")
svclassifier = SVC(kernel=f"{model}")
svclassifier.fit(X_train.values, y_train.values)
print("trained model.\n")

# -----------------------------------------------------------
print("making predictions...")
y_pred = svclassifier.predict(X_test)
prediction_file_name = "submission"


def create_submission(preds, index):
    with open(f"{prediction_file_name}.csv", "w+") as file:
        for i in range(48):
            file.write(f"{index[i]} {preds[i]}\n")


print("creating submission...")
create_submission(y_pred, indices)
print(f"predictions file: {prediction_file_name}.csv")
