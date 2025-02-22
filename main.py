import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import recall_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data
df = pd.read_csv('heart.csv')

# Splitting data
x, y = df.drop('target', axis=1), df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=9)

# Scale-insensitive models
forest = RandomForestClassifier()
forest.fit(x_train, y_train)

nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)

gb_clf = GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)

# Scale-sensitive models
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y_train)

log = LogisticRegression()
log.fit(x_train_scaled, y_train)

svc = SVC()
svc.fit(x_train_scaled, y_train)

# Model Scores
print("Random Forest Score:", forest.score(x_test, y_test))
print("Naive Bayes Score:", nb_clf.score(x_test, y_test))
print("Gradient Boosting Score:", gb_clf.score(x_test, y_test))
print("KNN Score:", knn.score(x_test_scaled, y_test))
print("Logistic Regression Score:", log.score(x_test_scaled, y_test))
print("SVC Score:", svc.score(x_test_scaled, y_test))

# Recall Scores
print('Random Forest Recall:', recall_score(y_test, forest.predict(x_test)))
print('Naive Bayes Recall:', recall_score(y_test, nb_clf.predict(x_test)))
print('Gradient Boosting Recall:', recall_score(y_test, gb_clf.predict(x_test)))
print('KNN Recall:', recall_score(y_test, knn.predict(x_test_scaled)))
print('Logistic Regression Recall:', recall_score(y_test, log.predict(x_test_scaled)))
print('SVC Recall:', recall_score(y_test, svc.predict(x_test_scaled)))

# ROC Curve
y_probs = forest.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.show()

print("ROC AUC Score:", roc_auc_score(y_test, y_probs))

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

forest = RandomForestClassifier(n_jobs=-1, random_state=9)
grid_search = GridSearchCV(forest, param_grid, cv=3, n_jobs=-1)
grid_search.fit(x_train, y_train)

best_forest = grid_search.best_estimator_

# Feature Importances
feature_importances = best_forest.feature_importances_
features = np.array(best_forest.feature_names_in_)

sorted_idx = np.argsort(feature_importances)
sorted_features = features[sorted_idx]
sorted_importances = feature_importances[sorted_idx]

plt.figure(figsize=(10, 6))
colors = plt.cm.YlGn(sorted_importances / max(sorted_importances))
plt.barh(sorted_features, sorted_importances, color=colors)
plt.xlabel('Feature Importances')
plt.ylabel("Features")
plt.title('Feature Importance of Best Random Forest Model')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='YlGn', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Best model score
print("Best Random Forest Score:", best_forest.score(x_test, y_test))
