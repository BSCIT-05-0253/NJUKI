# Njuki
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load your historical maintenance data
# Assuming you have a dataset with features (X) and a target variable (y)
# X should include features like temperature, pressure, vibration, etc.
# y should be binary, indicating whether the equipment failed (1) or not (0)
# Modify this part according to your dataset structure

# Example:
# data = pd.read_csv('/content/machines sensors.csv')
# X = data.drop('sensor_1', axis=1)
# y = data['machine_failure']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and a random forest classifier
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
