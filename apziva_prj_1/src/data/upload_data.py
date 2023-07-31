import pandas as pd

def load_data(file_path):
    """Load the dataset from the given file path"""
    data = pd.read_csv(file_path)
    return data
