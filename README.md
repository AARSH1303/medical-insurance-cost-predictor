# 🩺 HealthSure AI — Medical Insurance Cost Predictor

An interactive end-to-end Machine Learning web application designed to forecast individual annual medical insurance charges using demographic data, body composition metrics, and lifestyle risk factors.

---

## 🛠️ Tech Stack
- **Language:** Python 3.11+
- **Data Manipulation & Preprocessing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Random Forest Regressor)
- **Model Serialization:** Joblib
- **Frontend / Web UI:** Streamlit

---

## 📊 Features & Input Variables
- `Age`: Primary beneficiary age
- `Sex`: Biological sex indicators (Female / Male)
- `BMI`: Body Mass Index ($kg/m^2$) calculated dynamically via height and weight inputs
- `Children`: Number of covered dependents
- `Smoker`: Tobacco usage status (Primary actuarial risk multiplier)
- `Region`: Geographic US census region (northeast, northwest, southeast, southwest)

---

## 🚀 How to Run Locally

1. **Clone Repository:**
   ```bash
   git clone [https://github.com/AARSH1303/medical-insurance-cost-predictor.git](https://github.com/AARSH1303/medical-insurance-cost-predictor.git)
   cd medical-insurance-cost-predictor


   python -m venv venv
.\venv\Scripts\activate
pip install pandas numpy scikit-learn streamlit joblib notebook



python -m streamlit run app.py