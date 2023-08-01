import matplotlib.pyplot as plt
import seaborn as sns
from upload_data import data
from predict_model import y_pred

def plot_actual_vs_predicted(y_true, y_pred):
    """Plot the difference between actual and predicted values using Seaborn."""
    plt.figure(figsize=(5, 5))

    ax = sns.distplot(y_true, hist=False, color="orange", label="Actual Value")
    sns.distplot(y_pred, hist=False, color="green", label="Predicted Values", ax=ax)

    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()
    plt.close()

plot_actual_vs_predicted(data["Y"], y_pred)
