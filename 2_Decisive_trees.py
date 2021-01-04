import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix

# 2.2.2
# cats = pd.read_csv('https://stepik.org/media/attachments/course/4852/cats.csv')
# result = [-(1/1) * np.log2(1/1) - 0, -(4/9) * np.log2(4/9)-(5/9) * np.log2(5/9),
#           0-(5/5) * np.log2(5/5), -(4/5)*np.log2(4/5) - (1/5)*np.log2(1/5),
#           0 - (6/6)*np.log2((6/6)), -(4/4)*np.log2((4/4)) - 0]
# for i in result:
#     print(round(i,2))

# 2.2.3
# cats = pd.read_csv('https://stepik.org/media/attachments/course/4852/cats.csv')
# eshki = [-(1/1) * np.log2(1/1) - 0, -(4/9) * np.log2(4/9)-(5/9) * np.log2(5/9),
#           0-(5/5) * np.log2(5/5), -(4/5)*np.log2(4/5) - (1/5)*np.log2(1/5),
#           0 - (6/6)*np.log2((6/6)), -(4/4)*np.log2((4/4)) - 0]
#
# ee = -(4/10)*np.log2(4/10) - (6/10)*np.log2(6/10)
# result = [ee - (1/10)*eshki[0] - (9/10)*eshki[1],
#           ee - (5/10)*eshki[2] - (5/10)*eshki[3],
#           ee - (6/10)*eshki[4] - (6/10)*eshki[5]]
# for i in result:
#     print(round(i,2))

# 2.4.3
# train = pd.read_csv('https://stepik.org/media/attachments/course/4852/train_iris.csv')
# test = pd.read_csv('https://stepik.org/media/attachments/course/4852/test_iris.csv')
# x_train = train.drop(['Unnamed: 0','species'], axis= 1)
# y_train = train.species
# x_test = test.drop(['Unnamed: 0','species'], axis= 1)
# y_test = test.species
# max_depth_values = range(1,100)
# scores_data = pd.DataFrame()
# for max_depth in max_depth_values:
#     clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
#     clf.fit(x_train,y_train)
#     train_score = clf.score(x_train,y_train)
#     test_score = clf.score(x_test, y_test)
#     temp_score_data = pd.DataFrame({'max_depth':[max_depth],'train_score':[train_score],'test_score':[test_score]})
#     scores_data = scores_data.append(temp_score_data)
#
# scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score','test_score'],
#                            var_name='set_type', value_name='score')
# sns.lineplot(x='max_depth', y='score', hue='set_type', data = scores_data_long)
# plt.show()

# 2.4.4
# train = pd.read_csv('https://stepik.org/media/attachments/course/4852/dogs_n_cats.csv')
# x_train = train.drop('Вид', axis= 1)
# y_train = train.Вид
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf.fit(x_train,y_train)
# test = pd.read_json('dataset_209691_15.txt')
# pred = list(clf.predict(test))
# print(pred.count('собачка'))

# 2.4.5
# clf = DecisionTreeClassifier()
# clf.fit(X_train,y_train)
# predictions = clf.predict(X_test)
# precision = precision_score(y_test, predictions, average='micro')

# 2.7.1
# dt = DecisionTreeClassifier(max_depth=5, min_samples_split = 5)

# 2.7.2
# train = pd.read_csv('https://stepik.org/media/attachments/course/4852/train_data_tree.csv')
# x_train = train.drop('num', axis= 1)
# y_train = train.num
# clf = DecisionTreeClassifier(criterion='entropy')
# clf.fit(x_train, y_train)
# tree.plot_tree(clf, filled=True)
# plt.show()
# l_node = clf.tree_.children_left[0]
# r_node = clf.tree_.children_right[0]
# n0 = clf.tree_.n_node_samples[l_node]
# e0 = clf.tree_.impurity[l_node]
# n1 = clf.tree_.n_node_samples[r_node]
# e1 = clf.tree_.impurity[r_node]
# n = n0+n1
# ig = round(0.996 - (n0*e0 + n1*e1)/n,3)
# print(ig)

# 2.7.5
# iris = load_iris()
# X = iris.data
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, train_size=0.75)
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# predicted = dt.predict(X_test)

# 2.7.8
# iris = load_iris()
# X = iris.data
# y = iris.target
# dt = DecisionTreeClassifier()
# parametrs = {'max_depth' : range(1,10), 'min_samples_split' : range(2,10), 'min_samples_leaf' : range(1,10)}
# search = GridSearchCV(dt, parametrs, cv=5)
# search.fit(X, y)
# best_tree = search.best_estimator_

# 2.7.9
# iris = load_iris()
# X = iris.data
# y = iris.target
# dt = DecisionTreeClassifier()
# parametrs = {'max_depth' : range(1,10), 'min_samples_split' : range(2,10), 'min_samples_leaf' : range(1,10)}
# search = RandomizedSearchCV(dt, parametrs)
# search.fit(X, y)
# best_tree = search.best_estimator_

# 2.7.10
# x_train = train.drop('y', axis= 1)
# y_train = train.y
# dt = DecisionTreeClassifier()
# parametrs = {'max_depth' : range(1,10), 'min_samples_split' : range(2,10), 'min_samples_leaf' : range(1,10)}
# search = GridSearchCV(dt, parametrs, cv=5)
# search.fit(x_train, y_train)
# best_tree = search.best_estimator_
# predictions = best_tree.predict(test)

# 2.7.11
# conf_matrix = confusion_matrix(y, predictions)

# 2.10.1
# submissions = pd.read_csv('submissions_data_train.csv')
# only_wrong = submissions[submissions['submission_status']=='wrong']
# print(only_wrong.step_id.value_counts().head(1))