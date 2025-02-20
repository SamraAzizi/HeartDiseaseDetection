import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


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