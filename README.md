Steel Plate Fault Classification

1. Project Overview

This project implements and compares five Machine Learning classification algorithms to identify different types of faults in steel plates.

The objective is to evaluate multiple classification models and identify the most suitable model using standard performance metrics.

2. Dataset

- Dataset: Steel Plates Faults
- Source: UCI Machine Learning Repository
- Dataset ID: 198
- Total instances: 1941
- Input features: 27
- Target classes: 7
- Missing values: 0

Fault Categories

1. Bumps
2. Dirtiness
3. K_Scratch
4. Other_Faults
5. Pastry
6. Stains
7. Z_Scratch

3. Data Preprocessing

- Checked dataset dimensions and missing values.
- Converted one-hot encoded target columns into class labels.
- Split the dataset into 80% training and 20% testing.
- Used stratified sampling to preserve class proportions.
- Applied feature scaling for Logistic Regression, KNN and SVM.

Random state: 42

4. Machine Learning Models

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Each trained model was saved using Joblib for later inference.

5. Evaluation Metrics

The models were evaluated using:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Precision, Recall and F1 Score use macro averaging. AUC uses macro one-vs-rest evaluation.

6. Model Comparison

Model| Accuracy| AUC| Precision| Recall| F1 Score| MCC
Logistic Regression| 0.7275| 0.9379| 0.7613| 0.7305| 0.7418| 0.6487
Decision Tree| 0.7429| 0.8534| 0.7531| 0.7559| 0.7530| 0.6688
KNN| 0.7275| 0.9184| 0.7458| 0.7482| 0.7416| 0.6548
Gaussian Naive Bayes| 0.4524| 0.8386| 0.4315| 0.4795| 0.3830| 0.3789
SVM| 0.7506| 0.9493| 0.8006| 0.7390| 0.7626| 0.6765

7. Results and Observations

- SVM achieved the highest accuracy (75.06%), AUC (0.9493), precision (0.8006), F1 score (0.7626) and MCC (0.6765).
- Decision Tree achieved the highest recall (0.7559).
- Logistic Regression and KNN demonstrated competitive performance.
- Gaussian Naive Bayes showed comparatively lower performance.
- The dataset exhibits class imbalance, which can influence classification performance across different fault categories.

Based on the overall evaluation, SVM demonstrated the best overall performance among the five models.

8. Streamlit Web Application

The application allows users to:

- Upload a test dataset in CSV format.
- Select one of the five trained models.
- Evaluate the uploaded dataset.
- View all six evaluation metrics.
- View the confusion matrix.
- View the classification report.

The application loads previously saved models and does not retrain them during evaluation.

Live Application:
https://steel-plate-fault-classification.streamlit.app/

9. GitHub Repository

https://github.com/anandtanuku/Steel_Plate_Fault

10. Project Structure

Steel_Plate_Fault/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── results_table.csv
│
└── model/
    ├── Logistic_Regression.joblib
    ├── Decision_Tree.joblib
    ├── KNN.joblib
    ├── Naive_Bayes.joblib
    └── SVM.joblib

11. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

12. Conclusion

Five classification algorithms were implemented and evaluated on the Steel Plates Faults dataset.

SVM achieved the strongest overall performance, while Decision Tree achieved the highest recall. The trained models were deployed through a Streamlit application for interactive evaluation using uploaded test datasets.
