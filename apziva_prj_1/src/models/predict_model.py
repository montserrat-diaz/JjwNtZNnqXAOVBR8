from sklearn.ensemble import RandomForestClassifier

def predict_model(classifier, X_test):
    y_pred = classifier.predict(X_test)
    return y_pred
