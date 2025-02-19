import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('heart.csv')

x, y = df.drop('target', axis=1), df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=9)
