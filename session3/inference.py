import joblib
model = joblib.load('random_forest_model.joblib')
# input_data = [[5.1, 3.5, 1.4, 0.2]]
input_data = [[6.7,3.0,5.2,2.3]]
prediction = model.predict(input_data)
print("Prediction for input data {}: {}".format(input_data, prediction))