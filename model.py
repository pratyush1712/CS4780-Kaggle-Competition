import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

sc = StandardScaler()

# -----------------------------------------------------------
print("loading data...")
training_data = pd.read_csv("./train_data.csv")
testing_data = pd.read_csv("./test_data_s.csv")
print("data loaded.\n")
prediction_file_name = "submission"


def submission():
    # -----------------------------------------------------------
    print("processing data...")
    # drop the tables that won't be needed: X - Category, Index
    X_train = training_data.drop("Group (0-Normal Control, 1 Affected)", axis=1)
    X_train = X_train.drop("Index ", axis=1)
    y_train = training_data["Group (0-Normal Control, 1 Affected)"]

    # scale the data
    X_train = sc.fit_transform(X_train)
    indices = testing_data["Index "]
    X_test = testing_data.drop("Index ", axis=1)

    # fill empty values of testing data with the mean values of those features
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp = imp.fit(X_test)
    X_test = imp.transform(X_test)
    X_test = sc.transform(X_test)
    print("data processed.\n")

    # -----------------------------------------------------------
    # specify the model used below
    model = "sigmoid"
    print(f"training {model} model...")
    svclassifier = SVC(kernel=f"{model}")
    svclassifier.fit(X_train, y_train.values)
    print("trained model.\n")

    # -----------------------------------------------------------
    print("making predictions...")
    y_pred = svclassifier.predict(X_test)

    print("creating submission...")
    create_submission(y_pred, indices)
    print(f"predictions file: {prediction_file_name}.csv")


def create_submission(preds, index):
    with open(f"{prediction_file_name}.csv", "w+") as file:
        file.write("Index,Categories\n")
        for i in range(48):
            file.write(f"{index[i]},{preds[i]}\n")


def kfold():
    X = training_data.drop("Group (0-Normal Control, 1 Affected)", axis=1)
    X = X.drop("Index ", axis=1)
    y = training_data["Group (0-Normal Control, 1 Affected)"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.10, shuffle=True
    )
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    svclassifier = SVC(kernel="sigmoid")
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_val)
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))


submission()
