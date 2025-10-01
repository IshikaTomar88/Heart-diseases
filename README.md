# ❤️ Heart Disease Prediction

## 📌 Overview  
This project aims to **predict the presence of heart disease** in patients using machine learning models.  
By analyzing key health indicators such as age, cholesterol levels, blood pressure, and chest pain type,  
the notebook demonstrates how data-driven models can support early diagnosis and healthcare decision-making.  

---

## 📂 Dataset  
- Source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) (also available on Kaggle).  
- Features: medical attributes like age, sex, chest pain type, blood pressure, cholesterol, fasting blood sugar, ECG, etc.  
- Target variable: **`target`** (1 = presence of heart disease, 0 = absence).  

---

## ⚙️ Steps Followed  

### 1. Import Libraries  
- `numpy`, `pandas` → data handling  
- `matplotlib`, `seaborn` → data visualization  
- `scikit-learn` → preprocessing, model training & evaluation  

---

### 2. Data Exploration (EDA)  
- Inspected dataset shape, summary, and null values.  
- Visualized distributions of features such as **age, cholesterol, chest pain type, and resting BP**.  
- Explored correlation between features and the target variable.  

📸 Example Correlation Heatmap:  
![Correlation Heatmap](images/heart_correlation.png)  

---

### 3. Data Preprocessing  
- Converted categorical variables (e.g., chest pain type) into numerical form.  
- Standardized continuous features such as cholesterol and resting BP.  
- Split dataset into **train** and **test** sets.  

---

### 4. Model Building  
Trained multiple classification models:  
- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Random Forest Classifier**  
- **K-Nearest Neighbors (KNN)**  
- **Support Vector Machine (SVM)**  

---

### 5. Model Evaluation  
- Metrics used: **Accuracy, Precision, Recall, F1-score, ROC-AUC**  
- Visualized confusion matrices for each model.  

📸 Example Confusion Matrix:  
![Confusion Matrix](images/heart_confusion_matrix.png)  

📸 Example ROC Curve:  
![ROC Curve](images/heart_roc_curve.png)  

---

## 📊 Results  
- **Random Forest and Logistic Regression** achieved the highest accuracy.  
- ROC-AUC curve showed strong predictive power for ensemble methods.  
- Features like **chest pain type, age, cholesterol, resting BP, and max heart rate** had the strongest influence on predictions.  

---

## 🚀 Future Improvements  
- Hyperparameter tuning using GridSearchCV / RandomizedSearchCV.  
- Try boosting models (XGBoost, LightGBM, CatBoost).  
- Deploy the model using **Streamlit** or **Flask** for real-world usability.  

---

## 📦 How to Run  
1. Clone the repo and install requirements:  
   ```bash
   pip install -r requirements.txt

2. Run the notebook:

       jupyter notebook Heart_Diseases.ipynb

3. Execute all cells to preprocess data, train models, and visualize outputs.

🙌 Acknowledgments

Dataset from UCI Machine Learning Repository.

Project inspired by applications of machine learning in healthcare.
