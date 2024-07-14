import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
mlflow.set_tracking_uri("http://localhost:5000")


db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

logged_model = 'runs:/acb3db6240d04329acdbfc0b91c61eca/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

predictions = loaded_model.predict(X_test[0:10])
print(predictions)

