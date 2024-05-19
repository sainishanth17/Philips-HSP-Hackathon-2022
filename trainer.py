# Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling

# Training Dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#EDA of training data
profile=dataset.profile_report(title='EDA of Training Data')
profile.to_file(output_file="eda_train.html")

# SciKit Learn Preprocessing Library Import
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# SciKit Learn Classifier Library Import
from sklearn.svm import SVC
svc = svm.SVC()
parameters = {'kernel':('linear','rbf'), 'C':[1,120]}
classifier=GridSearchCV(svc,parameters)
#classifier = SVC(kernel = ('linear','rbf'))
classifier.fit(X, y)

# Test dataset
test = pd.read_csv('test.csv')
test = test.drop('id',1)
X_test = test.iloc[:, :].values
X_test = sc.transform(X_test)

y_pred = classifier.predict(X_test)


test = pd.read_csv('test.csv')
i = pd.DataFrame(test, columns=['id'])
l = pd.DataFrame(y_pred, columns=['price_range'])
i.join(l)

res = pd.concat([i, l], axis=1)

res.to_csv('result.csv',encoding='utf-8', index=False)

result = pd.read_csv('result.csv')


