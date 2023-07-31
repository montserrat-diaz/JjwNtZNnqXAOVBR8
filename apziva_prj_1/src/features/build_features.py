import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    X = data.drop(["Y"], axis=1)
    y = data["Y"]

    # standardize the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test
