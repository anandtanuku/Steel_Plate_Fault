# Steel Plate Fault Classification

## 1. Problem Statement

Steel plates can have different surface and structural faults. This project builds and compares five machine learning classifiers that predict the fault type of a steel plate from 27 measured features, and identifies the best model using standard evaluation metrics.

## 2. Dataset Description

| Item | Detail |
|---|---|
| Dataset | Steel Plates Faults |
| Source | UCI Machine Learning Repository (ID 198) |
| Instances | 1941 |
| Input features | 27 |
| Target classes | 7 |
| Missing values | 0 |

**Fault categories:** Bumps, Dirtiness, K_Scratch, Other_Faults, Pastry, Stains, Z_Scratch

The original target is stored as 7 one-hot columns. These were converted into a single class label. The classes are imbalanced: Other_Faults has the most samples and Dirtiness the fewest.

## 3. GitHub Repository

[Steel_Plate_Fault](https://github.com/anandtanuku/Steel_Plate_Fault)

## 4. Data Preprocessing

- Checked dataset shape and missing values.
- Converted the one-hot target columns into one class label.
- Split the data 80% train and 20% test, stratified by class (random state 42).
- Applied feature scaling for Logistic Regression, KNN and SVM.

## 5. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Each trained model is saved with Joblib in the `model/` folder.

**Metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC. Precision, Recall and F1 use macro averaging. AUC uses one-vs-rest.

## 6. Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7275 | 0.9379 | 0.7613 | 0.7305 | 0.7418 | 0.6487 |
| Decision Tree | 0.7429 | 0.8534 | 0.7531 | 0.7559 | 0.7530 | 0.6688 |
| KNN | 0.7275 | 0.9184 | 0.7458 | 0.7482 | 0.7416 | 0.6548 |
| Naive Bayes | 0.4524 | 0.8386 | 0.4315 | 0.4795 | 0.3830 | 0.3789 |
| **Support Vector Machine** | **0.7506** | **0.9493** | **0.8006** | 0.7390 | **0.7626** | **0.6765** |

## 7. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.7275 but a strong AUC of 0.9379, so it ranks the classes well. It struggles on overlapping cases because its boundary is a straight line. |
| Decision Tree | Accuracy 0.7429 but AUC only 0.8534. A tree gives rough, stepwise probabilities, so its ranking of cases is weaker. |
| KNN | Accuracy 0.7275 and AUC 0.9184. It depends on distances, so scaled features matter and rare classes can be outvoted. |
| Naive Bayes | Lowest, with accuracy 0.4524 and MCC 0.3789. It assumes features are independent, but several here are strongly related. |
| Support Vector Machine | Best AUC (0.9493), F1 (0.7626) and MCC (0.6765), with accuracy 0.7506. Its kernel fits curved class boundaries well. |

**Overall winner for this dataset: SVM**, because it has the highest accuracy, AUC, F1 and MCC. The test set is small (about 389 samples), so close scores may change on another split.

## 8. Streamlit Web Application

The app lets the user:

- Upload a test dataset (CSV, with the target column).
- Select one of the five trained models from a dropdown.
- View all six evaluation metrics.
- View the confusion matrix and classification report.

The app loads the saved models and does not retrain them.

**Live app:** [steel-plate-fault-classification.streamlit.app](https://steel-plate-fault-classification.streamlit.app/)

## 9. Project Structure

```text
Steel_Plate_Fault/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── results_table.csv
└── model/
    ├── Logistic_Regression.joblib
    ├── Decision_Tree.joblib
    ├── KNN.joblib
    ├── Naive_Bayes.joblib
    └── SVM.joblib
```

## 10. Technologies Used

Python, NumPy, Pandas, Scikit-learn, Matplotlib, Joblib, Streamlit
