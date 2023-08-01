import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
