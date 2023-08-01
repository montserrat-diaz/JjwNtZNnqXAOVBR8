import sys
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
    
    # Load data set
    data = load_data("happinesssurvey3.1.csv")

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Create the classifier
    classifier = create_classifier(n_estimators=10, max_depth=2)

    # Train the model
    train_model(classifier, X_train, y_train)

    # Prediction on the test set
    y_pred = predict_model(classifier, X_test)

    plot_actual_vs_predicted(data["Y"], y_pred)
