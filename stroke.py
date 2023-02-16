# Import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
df = pd.read_excel('brain_stroke.xlsx')

# Exploratory data analysis
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Pre-processing data
# Encoding categorical data
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['ever_married'] = label_encoder.fit_transform(df['ever_married'])
df['work_type'] = label_encoder.fit_transform(df['work_type'])
df['Residence_type'] = label_encoder.fit_transform(df['Residence_type'])
df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'])

# Handling missing values
df = df.dropna()

# Standardizing data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.iloc[:, :-1])

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(scaled_data, df.iloc[:, -1], test_size=0.2, random_state=0)

# Building model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting results
y_pred = classifier.predict(X_test)

# Evaluating model
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ")
print(cm)
print("Classification report: ")
print(classification_report(y_test, y_pred))
