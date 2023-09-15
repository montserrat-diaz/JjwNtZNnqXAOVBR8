import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from upload_data import data

def preprocess_data(data, target):
  unknown_replacement = np.nan
  data.replace("unknown", unknown_replacement, inplace=True)

  columns_to_fill=["job", "education"]
  for column in columns_to_fill:
    mode_value = data[column].mode().iloc[0]
    data[column].fillna(mode_value, inplace=True)

  data = data.drop(["contact"], axis=1)
    
  encoded_data = pd.get_dummies(data, columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'month'])

  mapping = {'yes': 1, 'no': 0}
  features_to_encode = ['y']
  encoded_data[features_to_encode] = encoded_data[features_to_encode].replace(mapping)
  
  features_to_standardize = ['age', 'balance', 'day', 'duration', 'campaign']
  scaler = StandardScaler()
  data[features_to_standardize] = scaler.fit_transform(data[features_to_standardize])

  minority_len = len(data[data[target]==1])
  majority_indices = data[data[target] == 0].index
  np.random.seed(42) #fixed random seed for reproducibility
  random_majority_indices = np.random.choice(majority_indices, minority_len, replace=False)
  minority_indices = data[data[target] == 1].index
  under_sample_indices = np.concatenate([minority_indices, random_majority_indices])
  under_sample = data.loc[under_sample_indices]

  X = data.loc[:, data.columns!=target]
  y = data.loc[:, data.columns==target]

  X_train, X_temp, y_train, y_temp = train_test_split(
      X, y, test_size=0.3, random_state=42
  )
  X_val, X_test, y_val, y_test = train_test_split(
      X_temp, y_temp, test_size=0.5, random_state=42
  )
  return X_train, y_train, X_val, y_val, X_test, y_test
