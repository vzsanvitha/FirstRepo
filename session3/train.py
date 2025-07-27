import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset...")
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['target'])

print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train.values.ravel())

print("Saving the trained model...")
joblib.dump(model, 'random_forest_model.joblib')
