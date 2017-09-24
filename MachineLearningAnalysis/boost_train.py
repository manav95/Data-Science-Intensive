from numpy import loadtxt
import catboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from catboost import Pool
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
a = 0
for col in colNames:
    a += 1

X_train, y_train = train[colNames], train['binary']
model = catboost.CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')

cat_features = []
i = 0
for col in colNames:
    if (meeting[col].dtype == 'object' and col != 'imdb_score'):
       cat_features.append(i)
    i += 1

trainPool = Pool(X_train, y_train, cat_features)
model.fit(trainPool)
print(model)

X_test, y_test = test[colNames], test['binary']
print(y_test)
testPool = Pool(X_test, y_test, cat_features)
preds_class  = model.predict(testPool)
print(len(preds_class))

#predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, preds_class)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
