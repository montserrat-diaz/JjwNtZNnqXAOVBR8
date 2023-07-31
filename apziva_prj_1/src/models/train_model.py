from sklearn.ensemble import RandomForestClassifier

def train_model(classifier, X_train, y_train):
    """Train the model using the training sets."""
    classifier.fit(X_train, y_train)
