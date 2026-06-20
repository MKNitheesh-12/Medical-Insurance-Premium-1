# Medical Insurance Premium Prediction

A machine learning project that predicts an individual's medical insurance premium based on personal health and lifestyle attributes. The project covers exploratory data analysis (EDA), data preprocessing, and a comparison of multiple classification and regression models to find the best predictor of premium price.

## 📋 Overview

Health insurance providers price premiums based on a range of risk factors — age, pre-existing conditions, surgical history, and more. This project uses a real-world dataset of 986 individuals to explore which factors most strongly influence premium pricing, and builds several models to predict the premium price for a given individual.

## 📊 Dataset

The dataset (`Medicalpremium.csv`) contains **986 records** and **11 columns**, with no missing values.

| Column | Description |
|---|---|
| `Age` | Age of the individual |
| `Diabetes` | Whether the individual has diabetes (0/1) |
| `BloodPressureProblems` | Whether the individual has blood pressure issues (0/1) |
| `AnyTransplants` | Whether the individual has had any transplants (0/1) |
| `AnyChronicDiseases` | Whether the individual has any chronic diseases (0/1) |
| `Height` | Height in cm |
| `Weight` | Weight in kg |
| `KnownAllergies` | Whether the individual has known allergies (0/1) |
| `HistoryOfCancerInFamily` | Family history of cancer (0/1) |
| `NumberOfMajorSurgeries` | Number of major surgeries undergone |
| `PremiumPrice` | Target variable — the insurance premium price |

## 🔍 Exploratory Data Analysis

The notebook performs the following EDA steps:

- Summary statistics with `describe()` (including custom percentiles)
- Data type and null-value inspection
- Distribution of `PremiumPrice` via count plots
- Correlation matrix and heatmap across all features
- Age distribution histogram
- Pairwise feature relationships via `sns.pairplot`, colored by `PremiumPrice`

Key takeaway: `Age` and `NumberOfMajorSurgeries` show the strongest positive correlation with `PremiumPrice`.

## ⚙️ Data Preprocessing

1. **Train/test split** — 30 samples held out for testing (`train_test_split`, `random_state=42`)
2. **Feature scaling** — `StandardScaler` applied to normalize input features before model training

## 🤖 Models Trained

The following models were trained and evaluated on the same train/test split:

| Model | Library |
|---|---|
| K-Nearest Neighbors (KNN) | `sklearn.neighbors.KNeighborsClassifier` |
| Support Vector Machine (SVM) | `sklearn.svm.SVC` |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` |
| Linear Regression | `sklearn.linear_model.LinearRegression` |

Each classification model is evaluated using a confusion matrix, accuracy score, and classification report (precision, recall, F1-score). Linear Regression is evaluated using R² score.

## 🏆 Results

| Algorithm | Accuracy (%) |
|---|---|
| **Random Forest** | **86.67** |
| Decision Tree | 80.00 |
| SVM | 80.00 |
| Linear Regression | 76.15 |
| KNN | 63.33 |

A horizontal bar chart visualizes these results, sorted by accuracy.

**Random Forest performed best** among all models tested on this dataset.

## 🛠️ Tech Stack

- **Python 3**
- **pandas** & **numpy** — data manipulation
- **matplotlib** & **seaborn** — visualization
- **scikit-learn** — preprocessing, modeling, and evaluation

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

### Usage
1. Clone this repository
2. Place `Medicalpremium.csv` in the project directory (update the file path in the notebook if needed — it was originally run on Google Colab with the file under `/content/`)
3. Open and run `MedicalPremium.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab

```bash
jupyter notebook MedicalPremium.ipynb
```

## 📁 Project Structure
```
.
├── MedicalPremium.ipynb     # Main analysis & modeling notebook
├── Medicalpremium.csv        # Dataset (not included — add your own)
└── README.md
```

## 📈 Future Improvements
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for each model
- Treat `PremiumPrice` as a continuous target with regression-focused models (e.g., Random Forest Regressor, Gradient Boosting) rather than only classification
- Cross-validation for more robust performance estimates
- Feature engineering (e.g., BMI from Height/Weight)
- Address class imbalance in less common premium price brackets

## 📄 License
This project is open source and available for educational purposes.
