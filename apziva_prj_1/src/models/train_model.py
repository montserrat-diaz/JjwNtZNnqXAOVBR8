import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

def preprocess_and_split_data(X, y, test_size=0.25, random_state=42):
    # standardize the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
