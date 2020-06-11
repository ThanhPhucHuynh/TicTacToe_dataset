import numpy as np
import pandas as pd

from sklearn.tree import plot_tree

# read data from file csv
df = pd.read_csv('tic-tac-toe.data',
        names = ["V1", "V2", "V3", "V4", "V5", "V6", 
                                    "V7", "V8", "V9", "V10"],
                                    sep=",")
print(df)


df['V1'],v1 = pd.factorize(df['V1'], sort=True)
df['V2'],v2 = pd.factorize(df['V2'], sort=True)
df['V3'],v3 = pd.factorize(df['V3'], sort=True)
df['V4'],v4 = pd.factorize(df['V4'], sort=True)
df['V5'],v5 = pd.factorize(df['V5'], sort=True)
df['V6'],v6 = pd.factorize(df['V6'], sort=True)
df['V7'],v7 = pd.factorize(df['V7'], sort=True)
df['V8'],v8 = pd.factorize(df['V8'], sort=True)
df['V9'],v9 = pd.factorize(df['V9'], sort=True)
df['V10'],v10 = pd.factorize(df['V10'], sort=True)
# print(v1,v10)
className = [v10[0],v10[1]]

# print( className)
# print(df.info())
# print(df.describe())
feature_names = ['V1','V2','V3','V4', 'V5', 'V6', 'V7', 'V8', 'V9']
x = df[feature_names] # Features

y = df['V10']

print(x,y)

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)



# clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=80) # change this classifier and check the impact
clf = GaussianNB()

clf = clf.fit(x_train,y_train)

thucte = y_test
dubao = clf.predict(x_test)
print(thucte)
print("haha \n")
print(dubao)
from sklearn.metrics import confusion_matrix
cnf_matrix_gnb = confusion_matrix(thucte,dubao)
print(cnf_matrix_gnb)
print('accuracy = ',accuracy_score(thucte, dubao))

# from sklearn import metrics

# # use the model to make predictions with the test data
# y_pred = clf.predict(x_test)
# # how did our model perform?
# count_misclassified = (y_test != y_pred).sum()
# print('Misclassified samples: {}'.format(count_misclassified))
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print('Accuracy: {:.2f}'.format(accuracy))


# negative_test = np.array ([2, 2, 1, 2, 1, 0, 1, 0, 0])
# positive_test = np.array ([2, 2, 2, 1, 1, 0, 1, 0, 0])
# test_group = [negative_test, positive_test]
# y_pred = clf.predict(test_group)
# print(y_pred) # should give [0, 1]