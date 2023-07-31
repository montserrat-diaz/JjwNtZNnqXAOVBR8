from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.25, random_state=42
)

classifier = RandomForestClassifier(n_estimators=10, max_depth=2)
classifier.fit(X_train, y_train)