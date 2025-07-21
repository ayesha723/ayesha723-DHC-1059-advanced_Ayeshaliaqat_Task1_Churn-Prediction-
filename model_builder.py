import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# Make sure static/plots directory exists
os.makedirs("static/plots", exist_ok=True)

# Load CSV
df = pd.read_csv("F:/artificial intelligence/DEV INTERN M1/churn_app/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Print basic info
print("Shape of DataFrame:", df.shape)
print("\nFirst 5 rows:\n", df.head())

print("\n--- DataFrame Info ---")
print(df.info())

# Drop 'customerID' column
df = df.drop(columns=["customerID"])

# Show updated preview
print("\nFirst 2 Rows After Dropping 'customerID':\n", df.head(2))

# Show column names
print("\nColumns:\n", df.columns)

# Show unique values in key columns
print("\nUnique values in 'gender':", df["gender"].unique())
print("Unique values in 'SeniorCitizen':", df["SeniorCitizen"].unique())

# Define numerical features
numerical_features_list = ["tenure", "MonthlyCharges", "TotalCharges"]

# Print unique values for non-numeric columns
print("\n--- Unique values in non-numerical columns ---")
for col in df.columns:
    if col not in numerical_features_list:
        print(f"{col}: {df[col].unique()}")
        print("-" * 50)

# Check missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Show rows where 'TotalCharges' is blank
blank_rows = df[df["TotalCharges"] == " "]
print(f"\nRows where 'TotalCharges' is blank:\n{blank_rows}")
print(f"Total blank entries in 'TotalCharges': {len(blank_rows)}")

# Replace blank values with 0.0 and convert type
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0.0")
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Confirm changes
print("\n--- DataFrame Info After Fixing 'TotalCharges' ---")
print(df.info())

# Show target variable distribution
print("\nTarget column ('Churn') class distribution:\n", df["Churn"].value_counts())

# Show shape, columns, and preview again
print("\nFinal DataFrame shape:", df.shape)
print("\nFinal column names:\n", df.columns)
print("\nFirst 2 Rows (Final):\n", df.head(2))

# Describe numeric features
print("\nDescriptive statistics:\n", df.describe())

# Function to plot histogram with mean and median lines
def plot_histogram(df, column_name, save_path):
    plt.figure(figsize=(5, 3))
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")

    # Calculate mean and median
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()

    # Add vertical lines
    plt.axvline(col_mean, color="red", linestyle="--", label="Mean")
    plt.axvline(col_median, color="green", linestyle="-", label="Median")

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)   # ✅ Save to correct path
    plt.close()              # ✅ Prevent displaying during batch run


# Call the function for "tenure" column
plot_histogram(df, "tenure", "static/plots/hist_tenure.png")
plot_histogram(df, "MonthlyCharges", "static/plots/hist_monthly.png")
plot_histogram(df, "TotalCharges", "static/plots/hist_total.png")

# Correlation heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/plots/heatmap.png")
plt.close()

object_cols = df.select_dtypes(include="object").columns.to_list()
object_cols = ["SeniorCitizen"] + object_cols

for col in object_cols:
    if col in ["gender", "Contract"]:  # Save only those two for now
        plt.figure(figsize=(5, 3))
        sns.countplot(x=df[col])
        plt.title(f"Count Plot of {col}")
        plt.tight_layout()
        save_path = f"static/plots/count_{col.lower()}.png"
        plt.savefig(save_path)  # ✅ Save plot
        plt.close()


# Encode target again just in case
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
print(df["Churn"].value_counts())

# Encode categorical columns
object_columns = df.select_dtypes(include="object").columns
encoders = {}

for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Original y_train distribution:\n", y_train.value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("After SMOTE y_train distribution:\n", y_train_smote.value_counts())

# Train models with cross-validation
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
}

cv_scores = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name}")
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
    cv_scores[model_name] = scores
    print(f"{model_name} cross-validation accuracy: {np.mean(scores):.2f}")
    print("-" * 50)

# Final model (Random Forest)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote, y_train_smote)

# Test evaluation
y_test_pred = rfc.predict(X_test)
print("\n--- Test Accuracy ---")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Save model
model_data = {"model": rfc, "features_names": X.columns.tolist()}
with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

# Load model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]
print("\nLoaded model:", loaded_model)
print("Feature names:", feature_names)

# New sample input for prediction
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

input_data_df = pd.DataFrame([input_data])

# Load encoders and transform input
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

# Predict
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

# Output results
print("\nPrediction Result:", prediction)
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print("Prediction Probability:", pred_prob)

