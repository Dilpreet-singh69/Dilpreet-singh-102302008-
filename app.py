import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load data from a CSV file
csv_file_path = './synthetic_admission_data.csv'  # Replace with your actual file path
data = pd.read_csv(csv_file_path)

# Check the first few rows of the data to understand its structure
st.write(data.head())

# Step 2: Split the data into features (X) and target (y)
X = data[['GRE', 'GPA', 'University_Rating']]
y = data['Admission']

# Step 3: Visualize feature distributions to choose scaling method
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(data['GRE'], kde=True, ax=axes[0])
axes[0].set_title('GRE Distribution')

sns.histplot(data['GPA'], kde=True, ax=axes[1])
axes[1].set_title('GPA Distribution')

sns.histplot(data['University_Rating'], kde=True, ax=axes[2])
axes[2].set_title('University Rating Distribution')

plt.tight_layout()
st.pyplot(fig)

# Step 4: Choose the appropriate scaling method
scaling_method = st.selectbox(
    'Choose Scaling Method:',
    ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'None']
)

# Step 5: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply selected scaler
if scaling_method == 'StandardScaler':
    scaler = StandardScaler()
elif scaling_method == 'MinMaxScaler':
    scaler = MinMaxScaler()
elif scaling_method == 'RobustScaler':
    scaler = RobustScaler()
else:
    scaler = None

if scaler:
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Step 6: Train models and evaluate
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC(probability=True)  # Ensure SVM outputs probabilities
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
        
    return results, models

# Train models and get accuracy results
results, models = train_and_evaluate(X_train, X_test, y_train, y_test)
st.write("Model Accuracy Results: ", results)

# Save the trained models and scaler
joblib.dump(models, 'all_models.pkl')  # Save all models
joblib.dump(scaler, 'scaler.pkl')

# Step 7: Streamlit app
# Load the trained scaler
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title('Student Admission Prediction')

# Dropdown to select the algorithm
model_choice = st.selectbox(
    'Select Prediction Algorithm:',
    ['Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM']
)

# Load the corresponding model based on the selection
models = joblib.load('all_models.pkl')
selected_model = models[model_choice]

# Inputs from the user
gre = st.slider('GRE Score', 260, 340, 300)
gpa = st.slider('GPA Score', 2.5, 4.0, 3.5)
university_rating = st.selectbox('University Rating', [1, 2, 3, 4, 5])

# Prediction
if st.button('Predict Admission'):
    input_data = np.array([[gre, gpa, university_rating]])
    
    # Scale the input data
    if scaler:
        input_data = scaler.transform(input_data)
    
    # Predict using the selected model
    prediction = selected_model.predict(input_data)  # Get the final prediction (0 or 1)
    
    # Check the predicted probability (for debugging purposes)
    prediction_proba = selected_model.predict_proba(input_data)  # Probability output
    
    st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f}")  # Probability of admission
    
    if prediction == 1:
        st.write("Congratulations! You have a chance of getting admitted.")
    else:
        st.write("Sorry, it seems you may not get admitted.")

# Step 8: Evaluate Model Performance
# Show confusion matrix and classification report on the test data
y_pred = selected_model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(conf_matrix)

# Classification Report
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))
