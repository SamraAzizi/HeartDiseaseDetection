import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier


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