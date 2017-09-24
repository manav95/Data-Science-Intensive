from numpy import loadtxt
import catboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

meeting = pd.read_csv('mergeWithRevenue.csv', encoding='latin-1')
meeting = meeting.fillna('')
seed = 7
train_size = .8

df = pd.DataFrame(np.random.randn(len(meeting), 2))
msk = np.random.rand(len(df)) < train_size
train = meeting[msk]
test = meeting[~msk]

colNames = [col for col in meeting.columns.values if col not in ['Unnamed: 0.1', 'Unnamed: 0', 'id', 'imdb_id', 'Rating', 'Unnamed: 0_y', 'binary', 'domgross',	'intgross',	'budget_2013$','domgross_2013$','intgross_2013$','box_office','adjusted_box_office']]
print(colNames)
X_train, y_train = train[colNames], train['binary']
model = catboost.CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')
cat_features = [0, 2, 3, 8, 11, 12, 15, 17, 18, 20, 21, 22, 28, 29, 30, 31, 32, 33, 34, 38, 39, 40]
model.fit(X_train, y_train, cat_features)
print(model)

X_test, y_test = test[colNames], test['binary']
y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
