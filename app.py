from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")

app = Flask(__name__)
os.makedirs("static/plots", exist_ok=True)

# Load model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
features = model_data["features_names"]

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Plotting functions
def plot_histogram(df, column_name):
    plt.figure(figsize=(5, 3))
    sns.histplot(df[column_name], kde=True)
    plt.axvline(df[column_name].mean(), color="red", linestyle="--", label="Mean")
    plt.axvline(df[column_name].median(), color="green", linestyle="-", label="Median")
    plt.legend()
    path = f"static/plots/hist_{column_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def plot_heatmap(df):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    path = "static/plots/heatmap.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def plot_countplot(df, column_name):
    plt.figure(figsize=(5, 3))
    sns.countplot(x=df[column_name])
    plt.title(f"Count Plot of {column_name}")
    path = f"static/plots/count_{column_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    message = ""

    if request.method == "POST":
        input_data = {col: request.form[col] for col in request.form}
        input_df = pd.DataFrame([input_data])

        input_df["SeniorCitizen"] = int(input_df["SeniorCitizen"])
        input_df["tenure"] = int(input_df["tenure"])
        input_df["MonthlyCharges"] = input_df["MonthlyCharges"].astype(float)
        input_df["TotalCharges"] = input_df["TotalCharges"].astype(float)


        for col, encoder in encoders.items():
            input_df[col] = encoder.transform(input_df[col])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        message = "⚠️ Likely to Churn. Offer 30% discount!" if prediction == 1 else "✅ Customer is likely to stay."

    df = pd.read_csv("F:/artificial intelligence/DEV INTERN M1/churn_app/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0.0").astype(float)

    plots = {
    "hist_tenure": "plots/hist_tenure.png",
    "hist_monthly": "plots/hist_monthly.png",
    "hist_total": "plots/hist_total.png",
    "heatmap": "plots/heatmap.png",
    "count_gender": "plots/count_gender.png",
    "count_contract": "plots/count_contract.png"
}



    return render_template("index.html", prediction=prediction, probability=probability, message=message, plots=plots)


if __name__ == "__main__":
    app.run(debug=True)
