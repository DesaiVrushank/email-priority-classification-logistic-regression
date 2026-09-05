# Email Priority Classification Using Logistic Regression

A binary classification Machine Learning project that predicts whether an incoming email should be marked as **High Priority** or **Normal/Low Priority** using Logistic Regression.

## Project Overview

With a large number of emails received every day, it can be difficult to identify which emails require immediate attention. Machine Learning can help automate this process by learning patterns from previously classified emails.

This project implements a **Logistic Regression** model to classify emails into two categories:

* `0` → Normal / Low Priority
* `1` → High Priority

The project covers the complete basic Machine Learning workflow, including data inspection, exploratory data analysis, data quality checking, preprocessing, model training, prediction, evaluation, error analysis, and Logistic Regression coefficient interpretation.

## Problem Statement

**Problem 08 — Email Priority Classification**

The objective is to build a binary classification solution that predicts whether an incoming email should be marked as high priority.

## Objective

The main objectives of this project are:

* Analyze the email priority dataset.
* Inspect data types and dataset structure.
* Check for missing values and duplicate records.
* Analyze target class distribution.
* Perform exploratory data analysis.
* Check feature distributions and correlations.
* Separate input features and target variable.
* Split the dataset into training and testing sets.
* Apply numerical preprocessing.
* Train a Logistic Regression classification model.
* Generate class and probability predictions.
* Evaluate the model using classification metrics.
* Analyze the confusion matrix and ROC curve.
* Interpret Logistic Regression coefficients and odds ratios.
* Perform error analysis and identify possible improvements.

## Dataset

The dataset used in this project is:

`dataset_08_email_priority_classification.csv`

### Dataset Details

* Total Records: **1,000**
* Total Columns: **7**
* Input Features: **6**
* Target Variable: **1**
* Missing Values: **0**
* Duplicate Rows: **0**

### Features

| Feature                   | Description                                        |
| ------------------------- | -------------------------------------------------- |
| `sender_frequency`        | Frequency of emails received from the sender       |
| `keyword_score`           | Priority-related keyword score                     |
| `thread_length`           | Number/length of messages in the email thread      |
| `response_deadline_hours` | Available time before the response deadline        |
| `attachment_count`        | Number of attachments                              |
| `previous_priority_rate`  | Historical rate of priority emails from the sender |

### Target

| Value | Meaning               |
| ----: | --------------------- |
|   `0` | Normal / Low Priority |
|   `1` | High Priority         |

The target classes are perfectly balanced:

* Normal / Low Priority: **500**
* High Priority: **500**

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab / Jupyter Notebook

## Machine Learning Algorithm

### Logistic Regression

Logistic Regression was selected because this is a **binary classification problem**.

The model estimates the probability that an email belongs to the High Priority class.

The classification output is:

```text
0 → Normal / Low Priority
1 → High Priority
```

## Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Missing Value Analysis
   ↓
Duplicate Check
   ↓
Class Balance Analysis
   ↓
Data Quality Check
   ↓
Exploratory Data Analysis
   ↓
Correlation Analysis
   ↓
Feature & Target Separation
   ↓
Train-Test Split
   ↓
Data Preprocessing
   ↓
Logistic Regression
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Confusion Matrix
   ↓
ROC Curve
   ↓
Coefficient & Odds Ratio Analysis
   ↓
Error Analysis
   ↓
Conclusion
```

## Exploratory Data Analysis

The project performs exploratory analysis to understand the dataset before model training.

The analysis includes:

* Dataset structure inspection
* Statistical summary
* Missing value analysis
* Duplicate detection
* Target class distribution
* Feature distributions
* Data quality checks
* Correlation matrix
* Correlation heatmap

## Data Preprocessing

The dataset contains numerical features with different ranges.

For example:

* `attachment_count` ranges from 0 to 10.
* `keyword_score` ranges from 0 to 100.
* `previous_priority_rate` ranges from 0 to 100.

Therefore, feature scaling is applied before Logistic Regression.

The preprocessing Pipeline contains:

```text
SimpleImputer
      ↓
StandardScaler
      ↓
Logistic Regression
```

### SimpleImputer

Median imputation is included to handle potential missing numerical values.

Although the current dataset contains no missing values, including the imputer makes the preprocessing pipeline more robust.

### StandardScaler

StandardScaler standardizes the numerical features so that they are on comparable scales.

## Train-Test Split

The dataset is divided using an **80:20 train-test split**.

```text
Training Data → 800 records
Testing Data  → 200 records
```

`random_state=42` is used for reproducibility.

Stratified sampling is used to maintain the target class distribution in both training and testing datasets.

### Training Distribution

```text
Class 0 → 400
Class 1 → 400
```

### Testing Distribution

```text
Class 0 → 100
Class 1 → 100
```

## Model Training

The Logistic Regression model is trained using the training dataset.

Model configuration:

| Parameter          | Value                              |
| ------------------ | ---------------------------------- |
| Algorithm          | Logistic Regression                |
| Test Size          | 20%                                |
| Random State       | 42                                 |
| Maximum Iterations | 1000                               |
| Class Weight       | None                               |
| Preprocessing      | Median Imputation + StandardScaler |

Class weighting was not required because the dataset is perfectly balanced.

## Model Prediction

Two types of predictions are generated.

### Class Prediction

The model predicts:

```text
0 → Normal / Low Priority
1 → High Priority
```

### Probability Prediction

The model also generates the probability of an email belonging to the High Priority class.

These probabilities are used for ROC curve and ROC-AUC analysis.

## Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Classification Report
* ROC Curve

## Model Performance

The final model achieved the following results on the test dataset:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **66.00%** |
| Precision | **64.81%** |
| Recall    | **70.00%** |
| F1-Score  | **67.31%** |
| ROC-AUC   | **73.43%** |

### Interpretation

The model correctly classified 66% of the test observations.

The recall of 70% means that the model correctly identified 70 out of 100 actual high-priority emails in the test dataset.

The ROC-AUC score of 0.7343 indicates that the model has a reasonable ability to distinguish between high-priority and normal/low-priority emails.

## Confusion Matrix

The model produced the following confusion matrix:

```text
[[62 38]
 [30 70]]
```

Therefore:

| Measure         | Value |
| --------------- | ----: |
| True Negatives  |    62 |
| False Positives |    38 |
| False Negatives |    30 |
| True Positives  |    70 |

### Interpretation

* **62** normal/low-priority emails were correctly classified.
* **38** normal/low-priority emails were incorrectly classified as high priority.
* **30** high-priority emails were incorrectly classified as normal/low priority.
* **70** high-priority emails were correctly classified.

The false negative rate for actual high-priority emails in the test set was **30%**.

## ROC-AUC Analysis

The ROC curve was generated using the predicted probabilities.

The model achieved:

**ROC-AUC = 0.7343**

This indicates that the model has meaningful class-discrimination ability, although additional improvements would be required for a stronger real-world system.

## Logistic Regression Coefficient Analysis

The Logistic Regression coefficients were analyzed to understand the direction and relative magnitude of feature associations.

| Feature                   | Coefficient | Odds Ratio |
| ------------------------- | ----------: | ---------: |
| `attachment_count`        |      0.7035 |     2.0209 |
| `sender_frequency`        |      0.6705 |     1.9552 |
| `thread_length`           |      0.6427 |     1.9016 |
| `previous_priority_rate`  |     -0.3719 |     0.6895 |
| `keyword_score`           |     -0.4161 |     0.6596 |
| `response_deadline_hours` |     -0.4949 |     0.6096 |

Since the features were standardized, these coefficients represent the model's learned associations for approximately a one-standard-deviation increase in a feature while other predictors are held constant.

The coefficients should be interpreted as model associations, not causal relationships.

## Feature Effect Ranking

Based on the absolute magnitude of the standardized Logistic Regression coefficients:

1. `attachment_count`
2. `sender_frequency`
3. `thread_length`
4. `response_deadline_hours`
5. `keyword_score`
6. `previous_priority_rate`

## Error Analysis

The main prediction errors were:

### False Positives

The model incorrectly marked **38 normal/low-priority emails** as high priority.

This could result in unnecessary priority notifications.

### False Negatives

The model incorrectly marked **30 high-priority emails** as normal/low priority.

This is important because an actual high-priority email could be overlooked.

For a real-world email system, reducing false negatives may be an important objective.

## Limitations

The current project has several limitations:

1. The dataset contains only 1,000 records.
2. The model uses only six numerical features.
3. Actual email subject and body text are not included.
4. The model does not use advanced NLP techniques.
5. The model was evaluated using a single train-test split.
6. The model has a 30% false negative rate on actual high-priority emails in the test set.
7. The dataset is perfectly balanced, while real-world email priority datasets may be imbalanced.
8. Logistic Regression assumes a linear relationship in the log-odds between predictors and the target.
9. The standard classification threshold may not be optimal for every real-world use case.

## Possible Improvements

Future versions of the project could include:

* Larger and more realistic email datasets.
* Email subject and body text.
* TF-IDF text features.
* Natural Language Processing techniques.
* Word or sentence embeddings.
* Cross-validation.
* Hyperparameter tuning.
* Comparison with other classification algorithms.
* Threshold optimization.
* Business-cost-based evaluation.
* Model monitoring and retraining.

Possible models for comparison include:

* Decision Tree
* Random Forest
* Support Vector Machine
* Gradient Boosting
* Other suitable classification algorithms

## Key Concepts Learned

Through this project, the following concepts were practiced:

* Supervised Learning
* Binary Classification
* Exploratory Data Analysis
* Data Inspection
* Data Quality Checking
* Missing Value Analysis
* Duplicate Detection
* Target Class Distribution
* Feature Distribution
* Correlation Analysis
* Feature and Target Separation
* Train-Test Split
* Stratified Sampling
* Data Preprocessing
* Median Imputation
* Feature Scaling
* StandardScaler
* Machine Learning Pipeline
* Logistic Regression
* Probability Prediction
* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve
* False Positive Analysis
* False Negative Analysis
* Logistic Regression Coefficients
* Odds Ratios
* Model Interpretation
* Error Analysis
* Data Leakage Prevention
* Classification Threshold

## Project Files

```text
email-priority-classification-logistic-regression/
│
├── dataset_08_email_priority_classification.csv
├── Email_Priority_Classification_Using_Logistic_Regression.ipynb
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project

Open the notebook:

```text
Email_Priority_Classification_Using_Logistic_Regression.ipynb
```

### 3. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the notebook

Execute the notebook cells from beginning to end.

## Project Outcome

This project successfully demonstrates a complete binary classification workflow for email priority prediction using Logistic Regression.

The model achieved:

* **66.00% Accuracy**
* **64.81% Precision**
* **70.00% Recall**
* **67.31% F1-Score**
* **73.43% ROC-AUC**

The project also demonstrates model interpretation using Logistic Regression coefficients and odds ratios.

## Internship Project

This project was completed as part of the **Machine Learning Internship at Learn Depth Academy**.

**Problem:** 08 — Email Priority Classification

## Author

**Vrushank Desai**

Computer Engineering Student
Machine Learning Intern — Learn Depth Academy
