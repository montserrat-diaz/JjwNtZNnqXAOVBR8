import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from read_data import data

def preprocess_data(data, target):
  unknown_replacement = np.nan
  data.replace("unknown", unknown_replacement, inplace=True)

  columns_to_fill=["job", "education"]
  for column in columns_to_fill:
    mode_value = data[column].mode().iloc[0]
    data[column].fillna(mode_value, inplace=True)

  data = data.drop(["contact"], axis=1)
    
  encoded_data = pd.get_dummies(data, columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'month'])

  # mapping dictionary for encoding 'yes' and 'no' to 1 and 0
  mapping = {'yes': 1, 'no': 0}
  features_to_encode = ['y']
  encoded_data[features_to_encode] = encoded_data[features_to_encode].replace(mapping)

  return encoded_data
