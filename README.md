# Heart Disease Prediction

## Overview
This project implements various machine learning models to predict the presence of heart disease using a dataset. The models include Random Forest, Naive Bayes, Gradient Boosting, K-Nearest Neighbors (KNN), Logistic Regression, and Support Vector Classifier (SVC). The code also includes hyperparameter tuning for the Random Forest model and visualizations for model performance and feature importance.

## Requirements
To run this code, you will need the following Python packages:
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

You can install the required packages using pip:
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

# Code Explanation
1. Data Loading: The dataset is loaded using pandas.

```bash
df = pd.read_csv('heart.csv')
```

2. Data Splitting: The data is split into features (x) and target (y), followed by a train-test split.
```bash
x, y = df.drop('target', axis=1), df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=9)
```

3. Model Training: Several models are trained on the training data:
- Random Forest 
- Naive Bayes
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Classifier (SVC)


4. Model Evaluation: The models are evaluated using accuracy and recall scores.

```bash
print("Random Forest Score:", forest.score(x_test, y_test))
```

5. ROC Curve: The ROC curve is plotted for the Random Forest model, and the ROC AUC score is calculated.
```bash
plt.plot(fpr, tpr)
```

6. Hyperparameter Tuning: A grid search is performed to find the best hyperparameters for the Random Forest model.