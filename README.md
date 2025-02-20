# Heart Disease Prediction

## 📌 Project Overview
The Heart Disease Prediction project aims to predict the likelihood of heart disease based on medical attributes using machine learning algorithms. This project involves data analysis, feature engineering, model building, and evaluation to provide accurate predictions.

## 📊 Dataset Information
- **Source**: The dataset is sourced from the UCI Machine Learning Repository.
- **File**: `heart.csv`
- **Attributes**:
  - `age`: Age of the patient
  - `sex`: Gender (1 = Male, 0 = Female)
  - `cp`: Chest pain type (0-3)
  - `trestbps`: Resting blood pressure (in mm Hg)
  - `chol`: Serum cholesterol (in mg/dl)
  - `fbs`: Fasting blood sugar (>120 mg/dl, 1 = True, 0 = False)
  - `restecg`: Resting electrocardiographic results (0-2)
  - `thalach`: Maximum heart rate achieved
  - `exang`: Exercise-induced angina (1 = Yes, 0 = No)
  - `oldpeak`: ST depression induced by exercise relative to rest
  - `slope`: Slope of the peak exercise ST segment (0-2)
  - `ca`: Number of major vessels (0-3) colored by fluoroscopy
  - `thal`: Thalassemia (0-3)
  - `target`: Presence of heart disease (1 = Disease, 0 = No Disease)

## 🧰 Technologies Used
- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest, etc.)
- Matplotlib, Seaborn (Data Visualization)

## 📊 Project Structure
```
├── Heart Disease Prediction.ipynb  # Main notebook for analysis and modeling
├── heart.csv                       # Dataset
└── README.md                       # Project documentation
```

## 📈 Model Building Process
1. **Data Preprocessing**: Handle missing values, encode categorical data.
2. **Exploratory Data Analysis (EDA)**: Visualize data distributions and relationships.
3. **Feature Engineering**: Scale and normalize features if required.
4. **Model Training**: Use machine learning models such as:
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
5. **Model Evaluation**: Assess performance using:
   - Accuracy
   - Precision, Recall, F1-score
   - ROC Curve, AUC

## 🚀 How to Run the Project
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2. **Set Up the Environment**:
    Ensure you have Python installed. Install required libraries using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
    Open `Heart Disease Prediction.ipynb` and execute the cells.

## 📊 Results
- Achieved high accuracy in predicting heart disease.
- Identified key factors contributing to heart disease such as age, chest pain type, and exercise-induced angina.

## 📌 Future Enhancements
- Explore deep learning models for better accuracy.
- Deploy the model using Flask or FastAPI.
- Integrate real-time prediction through a web interface.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests. Any contributions, issues, and feature requests are welcome.


