import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Model Definition ---
class WaterQualityModel(nn.Module):
    def __init__(self, input_size, hidden1=64, hidden2=32, dropout=0.3, num_classes=3):
        super(WaterQualityModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.dropout = nn.Dropout(dropout)
        self.output_wqi = nn.Linear(hidden2, 1)
        self.output_quality = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        wqi = self.output_wqi(x)
        quality = self.output_quality(x)
        return wqi, quality

# --- Streamlit App ---
st.set_page_config(page_title="Water Quality Prediction App", layout="wide")
st.markdown("""
<div style='display:flex; align-items:center; justify-content:space-between;'>
    <h1 style='margin:0;'>💧 Water Quality Prediction App</h1>
    <p style='font-size:0.9rem; color:gray;'>Built with PyTorch • scikit-learn • Streamlit</p>
</div>
""", unsafe_allow_html=True)

# --- About Section ---
st.sidebar.markdown("""
### ℹ️ About this App
This app uses a custom-built, AI-powered multi-task deep learning model implemented using PyTorch. It simultaneously performs supervised regression and multi-class classification on water quality data derived from chemical test results.

The model consists of:
- Input layer with 7 chemical features
- Two fully connected hidden layers with ReLU activation
- Dropout (p=0.3) between layers for regularization
- Output 1: Single linear neuron for WQI prediction (regression)
- Output 2: Fully connected layer with softmax activation for classification

Training configuration:
- Optimizer: Adam with learning rate = 0.001
- Loss functions: MSE for WQI and CrossEntropyLoss for class
- Epochs: 100 (batched on full training set)

---

- Predicts Water Quality Index (WQI) — a continuous score learned through supervised regression using mean squared error (MSE) loss on scaled targets
- Classifies water into ordinal categories using softmax-activated logits and cross-entropy loss, mapping chemical properties to categorical safety labels `Excellent`, `Good`, `Poor`, `Unsuitable for Drinking`, etc.
- The model architecture is implemented in PyTorch using two hidden layers with ReLU activations and dropout regularization. Scikit-learn is used for preprocessing (StandardScaler, LabelEncoder, train-test split).
- Features: `pH`, `EC`, `TDS`, `Ca`, `Mg`, `Na`, `Cl`
""")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your water quality CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    column_mapping = {"Water_Quality": "Water Quality Classification"}
    for new_col, actual_col in column_mapping.items():
        if actual_col in df.columns:
            df[new_col] = df[actual_col]

    features = ['pH', 'EC', 'TDS', 'Ca', 'Mg', 'Na', 'Cl']
    required_columns = ['WQI', 'Water_Quality'] + features
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    X = df[features]
    y_wqi = df['WQI'].values.reshape(-1, 1) / 100
    y_quality = df['Water_Quality']

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    le_quality = LabelEncoder()

    X_scaled = scaler_x.fit_transform(X)
    y_wqi_scaled = scaler_y.fit_transform(y_wqi)
    y_quality_encoded = le_quality.fit_transform(y_quality)

    num_classes = len(le_quality.classes_)

    X_train, X_test, y_wqi_train, y_wqi_test, y_quality_train, y_quality_test = train_test_split(
        X_scaled, y_wqi_scaled, y_quality_encoded, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.FloatTensor(X_train)
    y_wqi_train_tensor = torch.FloatTensor(y_wqi_train)
    y_quality_train_tensor = torch.LongTensor(y_quality_train)

    input_size = len(features)
    model = WaterQualityModel(input_size, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_wqi = nn.MSELoss()
    criterion_quality = nn.CrossEntropyLoss()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Uploaded Data Preview")
        st.dataframe(df.head())
        st.write("📊 Columns detected:", df.columns.tolist())
        st.write("🏷 Detected classes:", list(le_quality.classes_))

    with col2:
        st.subheader("🔁 Model Training")
        with st.spinner("Training the model..."):
            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                wqi_out, quality_out = model(X_train_tensor)
                loss_wqi = criterion_wqi(wqi_out, y_wqi_train_tensor)
                loss_quality = criterion_quality(quality_out, y_quality_train_tensor)
                total_loss = loss_wqi + loss_quality
                total_loss.backward()
                optimizer.step()
        st.success("✅ Model training complete!")

    # Prediction
    if st.button("🔮 Predict on Test Set"):
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():
            wqi_pred, quality_pred = model(X_test_tensor)
            wqi_pred = scaler_y.inverse_transform(wqi_pred.numpy()).flatten()
            actual_wqi = scaler_y.inverse_transform(y_wqi_test).flatten()
            predicted_classes = torch.argmax(quality_pred, dim=1).numpy()
            predicted_labels = le_quality.inverse_transform(predicted_classes)
            actual_labels = le_quality.inverse_transform(y_quality_test)

        result_df = pd.DataFrame({
            "Actual Classification": actual_labels,
            "Predicted Classification": predicted_labels
        })

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📈 WQI Prediction Plot")
            fig, ax = plt.subplots()
            ax.scatter(actual_wqi, wqi_pred, color='blue', alpha=0.5)
            ax.plot([0, max(actual_wqi)], [0, max(actual_wqi)], 'r--', label='Perfect Prediction')
            ax.set_xlabel("Actual WQI")
            ax.set_ylabel("Predicted WQI")
            ax.set_title("Predicted vs Actual WQI")
            ax.legend()
            st.pyplot(fig)

        with col4:
            st.subheader("🧪 Sample Predictions")
            st.dataframe(result_df.head(20))

            csv = result_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")
