import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- Streamlit UI ---
st.set_page_config(page_title="Water Quality Prediction App", layout="wide")
st.markdown("""
<div style='display:flex; align-items:center; justify-content:space-between;'>
    <h1 style='margin:0;'>💧 Water Quality Prediction App</h1>
    <p style='font-size:0.9rem; color:gray;'>Built with PyTorch • scikit-learn • Streamlit</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
### ℹ️ About this App
This app demonstrates a multi-task deep learning model for analyzing water quality using chemical test data.

🧠 **Model Overview**
- Built using **PyTorch** with two hidden layers, ReLU activations, and dropout regularization.
- Implements **multi-task learning** to:
  - 🔹 Perform **regression** on Water Quality Index (WQI)
  - 🔹 Perform **multi-class classification** on categorical water quality (e.g., Good, Poor)

⚙️ **Training Pipeline**
- Features: `pH`, `EC`, `TDS`, `Ca`, `Mg`, `Na`, `Cl`
- Preprocessing:
  - `StandardScaler` to normalize numeric features
  - `LabelEncoder` to encode classification targets
- Train/Test split: 80/20 with stratified labels

📉 **Loss Strategy**
- Regression Head → `MSELoss` for predicting scaled WQI
- Classification Head → `CrossEntropyLoss` for categorical output
- Joint optimization using a single backprop pass

📦 **Outputs**
- 🧪 Water Quality Index prediction (real-valued)
- 🏷 Water Quality Category prediction (multi-class)
- 📊 Confidence scores & class probabilities
- 📈 Interactive WQI plot + confusion matrix + classification report

Perfect for showcasing **environmental AI**, **multi-output neural networks**, and **interpretability** in water resource monitoring.
""", unsafe_allow_html=True)


# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your water quality CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Optional remapping
    column_mapping = {"Water_Quality": "Water Quality Classification"}
    for new_col, actual_col in column_mapping.items():
        if actual_col in df.columns:
            df[new_col] = df[actual_col]

    features = ['pH', 'EC', 'TDS', 'Ca', 'Mg', 'Na', 'Cl']
    required = ['WQI', 'Water_Quality'] + features
    if any(col not in df.columns for col in required):
        st.error(f"Missing required columns: {set(required) - set(df.columns)}")
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

    input_size = len(features)
    model = WaterQualityModel(input_size, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_wqi = nn.MSELoss()
    criterion_quality = nn.CrossEntropyLoss()

    st.subheader("📝 Uploaded Data Preview")
    st.dataframe(df.head())
    st.write("🏷 Detected classes:", list(le_quality.classes_))

    if st.button("🚀 Train Model"):
        with st.spinner("Training the model..."):
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                x_tensor = torch.FloatTensor(X_train)
                y_wqi_tensor = torch.FloatTensor(y_wqi_train)
                y_class_tensor = torch.LongTensor(y_quality_train)
                out_wqi, out_class = model(x_tensor)
                loss = criterion_wqi(out_wqi, y_wqi_tensor) + criterion_quality(out_class, y_class_tensor)
                loss.backward()
                optimizer.step()
        st.success("✅ Training complete!")

    if st.button("🔮 Predict on Test Set"):
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():
            pred_wqi, pred_class_logits = model(X_test_tensor)
            wqi_pred = scaler_y.inverse_transform(pred_wqi.numpy()).flatten()
            actual_wqi = scaler_y.inverse_transform(y_wqi_test).flatten()
            predicted_classes = torch.argmax(pred_class_logits, dim=1).numpy()
            predicted_labels = le_quality.inverse_transform(predicted_classes)
            actual_labels = le_quality.inverse_transform(y_quality_test)
            probs = torch.softmax(pred_class_logits, dim=1).numpy()
            confidence_scores = np.max(probs, axis=1)

        result_df = pd.DataFrame(X_test, columns=features)
        result_df["Actual WQI"] = actual_wqi
        result_df["Predicted WQI"] = wqi_pred
        result_df["Actual Classification"] = actual_labels
        result_df["Predicted Classification"] = predicted_labels
        result_df["Confidence"] = confidence_scores
        prob_df = pd.DataFrame(probs, columns=[f"Prob_{cls}" for cls in le_quality.classes_])
        result_df = pd.concat([result_df, prob_df], axis=1)

        with st.expander("🔍 Filter Predictions"):
            show_wrong = st.checkbox("Show only misclassifications")
            min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.01)
            filtered_df = result_df.copy()
            if show_wrong:
                filtered_df = filtered_df[filtered_df["Actual Classification"] != filtered_df["Predicted Classification"]]
            filtered_df = filtered_df[filtered_df["Confidence"] >= min_conf]

        def highlight(row):
            return ['background-color: #ffdddd' if row['Actual Classification'] != row['Predicted Classification'] else '' for _ in row]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 WQI Prediction Plot")
            fig, ax = plt.subplots()
            ax.scatter(actual_wqi, wqi_pred, color='blue', alpha=0.5)
            ax.plot([0, max(actual_wqi)], [0, max(actual_wqi)], 'r--')
            ax.set_xlabel("Actual WQI")
            ax.set_ylabel("Predicted WQI")
            ax.set_title("Predicted vs Actual WQI")
            st.pyplot(fig)

        with col2:
            st.subheader("🧪 Predictions Table")
            st.dataframe(filtered_df.head(20).style.apply(highlight, axis=1))
            st.download_button("⬇️ Download Results", filtered_df.to_csv(index=False), "predictions.csv", "text/csv")

        st.subheader("📋 Classification Report")
        st.text(classification_report(actual_labels, predicted_labels))

        st.subheader("🧮 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        cm = confusion_matrix(actual_labels, predicted_labels, labels=le_quality.classes_)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le_quality.classes_,
                    yticklabels=le_quality.classes_,
                    ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)
