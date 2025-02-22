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

```bash
grid_search = GridSearchCV(forest, param_grid, cv=3, n_jobs=-1)
```

7. Feature Importance: The feature importances of the best Random Forest model are visualized.

```bash 
plt.barh(sorted_features, sorted_importances)
```

8. Correlation Heatmap: A heatmap of feature correlations is generated.
```bash
sns.heatmap(df.corr(), annot=True, cmap='YlGn', fmt='.2f')
```


# Results
The code outputs the accuracy and recall scores for each model, the ROC AUC score for the Random Forest model, and visualizations for feature importance and correlation heatmap. The best Random Forest model's score is also printed.


# Usage

To run the code, simply execute the script in a Python environment where the required packages are installed. Ensure that the heart.csv file is accessible.


```bash
python main.py
```

# Conclusion

This project demonstrates the application of various machine learning algorithms for predicting heart disease. The results can help in understanding the importance of different features and the effectiveness of different models in making predictions.