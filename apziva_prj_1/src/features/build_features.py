from sklearn.preprocessing import StandardScaler

data = pd.read_csv("happinesssurvey3.1.csv")

X = data.drop(["Y"], axis=1)
y = data["Y"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)