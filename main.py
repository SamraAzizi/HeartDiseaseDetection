import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('heart.csv')

x, y = df.drop('target', axis=1), df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=9)


##scale insenitive 

forest = RandomForestClassifier()
forest.fit(x_train, y_train)

nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)

gb_clf = GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)

#scale sensitive model 

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y_train)


log = LogisticRegression()
log.fit(x_train_scaled, y_train)


svc = SVC()
svc.fit(x_train_scaled, y_train)



forest.score(x_test, y_test)

nb_clf.score(x_test, y_test)

gb_clf.score(x_test, y_test)

knn.score(x_test_scaled, y_test)

log.score(x_test_scaled, y_test)

svc.score(x_test_scaled, y_test)


y_preds = forest.predict(x_test)
print('forest: ', recall_score(y_test, y_preds))


y_preds = nb_clf.predict(x_test)
print('NB ', recall_score(y_test, y_preds))


y_preds = gb_clf.predict(x_test)
print('GB', recall_score(y_test, y_preds))

y_preds = knn.predict(x_test_scaled)
print('KNN ', recall_score(y_test, y_preds))


y_preds = log.predict(x_test_scaled) 
print('log ', recall_score(y_test, y_preds))


y_preds = svc.predict(x_test_scaled)
print('SVC', recall_score(y_test, y_preds))



y_probs = forest.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.show()

roc_auc_score(y_test, y_probs)


param_grid = {
    'n_estimators ': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2,5,10],
    'min_sample_leaf': [1,2,4],
    'max_features': ['sqrt','log2', None]

}

forest = RandomForestClassifier(in_jobs =-1, random_state=9)
grid_search = GridSearchCV(forest, param_grid, cv=3)