from sklearn.ensemble import RandomForestClassifier

def create_classifier(n_estimators=10, max_depth=2):
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return classifier
