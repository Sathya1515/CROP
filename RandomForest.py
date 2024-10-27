import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Split data into features (X) and labels (y)
X = data.drop('label', axis=1)  # Features (N, P, K, temperature, humidity, pH, rainfall)
y = data['label']                # Labels (crop types)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(rf_classifier, 'crop_recommendation_model.joblib')
