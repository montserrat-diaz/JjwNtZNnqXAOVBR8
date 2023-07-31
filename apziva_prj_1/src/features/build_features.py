import pandas as pd

def extract_features(data):
    """Extract the features from the data."""
    X = data.drop(["Y"], axis=1)
    y = data["Y"]
    return X, y
