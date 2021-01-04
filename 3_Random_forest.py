# 3.2.2
# rf = RandomForestClassifier(n_estimators = 15, max_depth=5)
# rf.fit(x_train, y_train)
# predictions = rf.predict(x_test)

# 3.5.1
# train = pd.read_csv('https://stepik.org/media/attachments/course/4852/training_mush.csv')
# train = train.rename(columns = {'class':'cl'})
# x = train.drop('cl', axis= 1)
# y = train.cl
# parametrs = {'n_estimators' : range(10,50,10), 'max_depth' : range(1,12,2), 'min_samples_leaf' : range(1,7),
#              'min_samples_split':range(2,9,2)}
# rf = RandomForestClassifier(random_state=0)
# search = GridSearchCV(rf, parametrs, cv=3, n_jobs=-1)
# search.fit(x, y)
# best_forest = search.best_estimator_
# print(best_forest.get_params())

# 3.5.2
# train = pd.read_csv('https://stepik.org/media/attachments/course/4852/training_mush.csv')
# train = train.rename(columns = {'class':'cl'})
# x = train.drop('cl', axis= 1)
# y = train.cl
# parametrs = {'n_estimators' : range(10,50,10), 'max_depth' : range(1,12,2), 'min_samples_leaf' : range(1,7),
#              'min_samples_split':range(2,9,2)}
# rf = RandomForestClassifier(random_state=0)
# search = GridSearchCV(rf, parametrs, cv=3, n_jobs=-1)
# search.fit(x, y)
# best_forest = search.best_estimator_
# imp = pd.DataFrame(best_forest.feature_importances_, index=x.columns, columns=['importance'])
# imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
# plt.show()

# 3.5.3
# train = pd.read_csv('https://stepik.org/media/attachments/course/4852/training_mush.csv')
# train = train.rename(columns = {'class':'cl'})
# x = train.drop('cl', axis= 1)
# y = train.cl
# parametrs = {'n_estimators' : range(10,50,10), 'max_depth' : range(1,12,2), 'min_samples_leaf' : range(1,7),
#              'min_samples_split':range(2,9,2)}
# rf = RandomForestClassifier(random_state=0)
# search = GridSearchCV(rf, parametrs, cv=3, n_jobs=-1)
# search.fit(x, y)
# best_forest = search.best_estimator_
# test = pd.read_csv('https://stepik.org/media/attachments/course/4852/testing_mush.csv')
# print(Counter(best_forest.predict(test)))

# 3.5.4
# train = pd.read_csv('https://stepik.org/media/attachments/course/4852/training_mush.csv')
# train = train.rename(columns = {'class':'cl'})
# x = train.drop('cl', axis= 1)
# y = train.cl
# parametrs = {'n_estimators' : range(10,50,10), 'max_depth' : range(1,12,2), 'min_samples_leaf' : range(1,7),
#              'min_samples_split':range(2,9,2)}
# rf = RandomForestClassifier(random_state=0)
# search = GridSearchCV(rf, parametrs, cv=3, n_jobs=-1)
# search.fit(x, y)
# best_forest = search.best_estimator_
# test = pd.read_csv('https://stepik.org/media/attachments/course/4852/testing_mush.csv')
# y_pred = best_forest.predict(test)
# y_true = pd.read_csv('testing_y_mush.csv')
# conf_matr = confusion_matrix(y_true,y_pred)
# sns.heatmap(conf_matr, annot=True,cmap="Blues")
# plt.show()

# 3.5.5
# train = pd.read_csv('https://stepik.org/media/attachments/course/4852/invasion.csv')
# x = train.drop('class', axis= 1)
# y = train['class'].map({'transport' :  0,  'fighter' :  1,  'cruiser' : 2})
# parametrs = {'n_estimators' : range(10,50,10), 'max_depth' : range(1,12,2), 'min_samples_leaf' : range(1,7),
#              'min_samples_split':range(2,9,2)}
# rf = RandomForestClassifier(random_state=0)
# search = GridSearchCV(rf, parametrs, cv=3, n_jobs=-1)
# search.fit(x, y)
# best_forest = search.best_estimator_
# test = pd.read_csv('https://stepik.org/media/attachments/course/4852/operative_information.csv')
# print(Counter(best_forest.predict(test)))

# 3.5.6
# train = pd.read_csv('https://stepik.org/media/attachments/course/4852/invasion.csv')
# x = train.drop('class', axis= 1)
# y = train['class'].map({'transport' :  0,  'fighter' :  1,  'cruiser' : 2})
# parametrs = {'n_estimators' : range(10,50,10), 'max_depth' : range(1,12,2), 'min_samples_leaf' : range(1,7),
#              'min_samples_split':range(2,9,2)}
# rf = RandomForestClassifier(random_state=0)
# search = GridSearchCV(rf, parametrs, cv=3, n_jobs=-1)
# search.fit(x, y)
# best_forest = search.best_estimator_
# test = pd.read_csv('https://stepik.org/media/attachments/course/4852/operative_information.csv')
# imp = pd.DataFrame(best_forest.feature_importances_, index=x.columns, columns=['importance'])
# imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
# plt.show()

# 3.5.7
# train = pd.read_csv('https://stepik.org/media/attachments/course/4852/space_can_be_a_dangerous_place.csv')
# sns.heatmap(train.corr(), cmap="Purples")
# plt.show()

# 3.7.1
# df = pd.read_csv('https://stepik.org/media/attachments/course/4852/iris.csv')
# before = time()
# df.apply(np.mean)
# after = time()
# print(after - before)
# before = time()
# df.mean(axis=0)
# after = time()
# print(after - before)
# before = time()
# df.describe().loc['mean']
# after = time()
# print(after - before)
# before = time()
# df.apply('mean')
# after = time()
# print(after - before)

# 3.7.2
# total_birds = wintering.expanding().sum()