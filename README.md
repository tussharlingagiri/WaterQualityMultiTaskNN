# 💧 WaterQualityMultiTaskNN

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Framework](https://img.shields.io/badge/Built%20With-PyTorch%20%26%20Streamlit-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Try it now](https://img.shields.io/badge/Streamlit-App-%F0%9F%94%8D-blue?logo=streamlit)](https://tusshar-water-quality-prediction.streamlit.app/)

A multi-task deep learning model in **PyTorch** to predict the **Water Quality Index (WQI)** (regression) and classify water as `Excellent`, `Good`, `Poor`, etc. (multi-class classification) based on chemical parameters.

👉 **[Launch App](https://tusshar-water-quality-prediction.streamlit.app/)**

---

## 📁 Folder Structure

```
WaterQualityMultiTaskNN/
├── app.py                # Streamlit app (main entry point)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── LICENSE               # MIT License
├── notebook/             # Jupyter notebooks for model development
└── venv/                 # Python virtual environment (excluded from version control)
```

---

## 🚀 Features

* ✅ Multi-task AI model: regression + classification
* 🔢 Predicts Water Quality Index (WQI) as a continuous score
* 📊 Classifies water quality categories (`Excellent`, `Good`, `Poor`, etc.)
* 🧠 Built with PyTorch, scikit-learn, Streamlit
* 📈 Real-time results: confidence scores, plots, misclassifications
* 📤 Upload your own `.csv` and download predictions

---

## 🔬 Technical Summary

* **Input features**: `pH`, `EC`, `TDS`, `Ca`, `Mg`, `Na`, `Cl`
* **Architecture**: 2-layer MLP + dropout regularization
* **Output 1**: Regression (WQI), optimized via MSE loss
* **Output 2**: Classification, optimized via CrossEntropy loss
* **Training**: Adam optimizer, 100 epochs
* **Preprocessing**: StandardScaler + LabelEncoder
* **Evaluation**: WQI scatter plot, classification report, confusion matrix

---

## 🧪 Sample Input

```csv
pH,EC,TDS,Ca,Mg,Na,Cl,WQI,Water_Quality
7.1,300,200,45,18,12,25,81.5,Good
6.8,500,410,80,35,20,40,65.3,Poor
...
```

---

## ⚙️ Setup Instructions

```bash
# 1. Clone repo
git clone https://github.com/tussharlion/WaterQualityMultiTaskNN.git
cd WaterQualityMultiTaskNN

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit app
streamlit run app.py
```

---

## 🌍 Online Deployment

Deployed on **Streamlit Community Cloud**:
🌐 [https://tusshar-water-quality-prediction.streamlit.app](https://tusshar-water-quality-prediction.streamlit.app)

---

## 📊 Outputs

* 📈 WQI vs Predicted WQI Scatter Plot
* 🧪 Filtered prediction table (highlighting misclassifications)
* 📋 Classification Report (Precision, Recall, F1)
* 🧮 Confusion Matrix (heatmap)
* ⬇️ CSV Download: Results + Class Probabilities + Confidence

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Tusshar Lingagiri**
🧑‍🎓 3rd-Year Software Engineering @ University of Glasgow
🔗 [LinkedIn](https://www.linkedin.com/in/tussharlingagiri) • [GitHub](https://github.com/tussharlion)



