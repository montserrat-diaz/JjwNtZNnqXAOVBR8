# %% [markdown]
# #**Data Upload**

# %%
from google.colab import files

uploaded = files. upload()

# %% [markdown]
# #**Information About Data**

# %%
#data processing
import pandas as pd
import numpy as np

import csv
filename = "happinesssurvey3.1.csv"

#initializing the titiles and rows list
fields = []
rows = []
columns = []

#reading cvs file
with open(filename, 'r') as csvfile:
  #creating a csv reader object
  csvreader = csv.reader(csvfile)

  #extracting field names through first row
  fields = next(csvreader)

  #extracting each data row one by one
  for row in csvreader:
    rows.append(row)

  #extracting each data column one by one
  for col in csvreader:
    columns.append(col)

  #get total number of rows
  print("Total No. of Rows: %d"%(csvreader.line_num))

#printing the fields names
print('Field Names Are: '+', '.join(field for field in fields))

#printing first five rows
print('\nData:\n')
for row in rows[:126]:
  #parsing each column of a row
  for col in row:
    print("%10s"%col,end=" "),
  print('\n')

# %% [markdown]
# #**Random Forest Classification Algorithm**

# %%
# packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# load data set
data = pd.read_csv("happinesssurvey3.1.csv")

# split data into input and taget variable(s)
X = data.drop(["Y"], axis=1)
y = data["Y"]

# standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.25, random_state=42
)

# create the classifier
classifier = RandomForestClassifier(n_estimators=10, max_depth=2)

# train the model using the training sets
classifier.fit(X_train, y_train)

# prediction on the test set
y_pred = classifier.predict(X_test)

# %% [markdown]
# #**Model Accuracy**

# %%
# calculate model accuray
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))

# %% [markdown]
# #**Important Features**

# %%
# check Important features
feature_importances_df = pd.DataFrame(
    {"feature": list(X.columns), "importance": classifier.feature_importances_}
).sort_values("importance", ascending=False)

feature_importances_df

# %% [markdown]
# #**Conclusions**

# %% [markdown]
# **The most important questions are X5 and X3.** \
# **The least important questions are X2 and X1.** \


