import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 1.4.3
# titanic = pd.read_csv('https://stepik.org/media/attachments/course/4852/titanic.csv')
# print(titanic.shape)
# print(titanic.dtypes.value_counts())

# 1.5.1
# students_performance = pd.read_csv('https://stepik.org/media/attachments/course/4852/StudentsPerformance.csv')
# print(students_performance[(students_performance['writing score'] > 99) & (students_performance.gender == 'female')])

# 1.5.2
# students_performance = pd.read_csv('https://stepik.org/media/attachments/course/4852/StudentsPerformance.csv')
# print((students_performance['lunch'] == 'free/reduced').mean())

# 1.5.3
# students_performance = pd.read_csv('https://stepik.org/media/attachments/course/4852/StudentsPerformance.csv')
# print('free/reduced')
# print(students_performance[students_performance['lunch'] == 'free/reduced'].describe())
# print('standard')
# print(students_performance[students_performance['lunch'] == 'standard'].describe())

# 1.5.6
# selected_columns = df.filter(like='-')

# 1.6.1
# dota = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')
# print(dota.groupby('legs').describe())

# 1.6.2
# lupupa = pd.read_csv('https://stepik.org/media/attachments/course/4852/accountancy.csv')
# print(lupupa.groupby(['Executor', 'Type']).mean())

# 1.6.3
# dota = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')
# print(dota.groupby(['attack_type', 'primary_attr']).describe())

# 1.6.4
# concentrations = pd.read_csv('http://stepik.org/media/attachments/course/4852/algae.csv')
# mean_concentrations = concentrations.groupby('genus').mean()

# 1.6.5
# concentrations = pd.read_csv('http://stepik.org/media/attachments/course/4852/algae.csv')
# print(concentrations.groupby('genus').aggregate({'alanin':'describe'}).loc['Fucus'].loc[[('alanin','min'),
# ('alanin','mean'), ('alanin','max')]].round(2))

# 1.6.6
# concentrations = pd.read_csv('http://stepik.org/media/attachments/course/4852/algae.csv')
# print(concentrations.groupby('group').var())
# print(concentrations.groupby('group').count())

# 1.7.1
# df = pd.read_csv('https://stepik.org/media/attachments/course/4852/income.csv')
# df.plot()
# plt.show()
# sns.lineplot(data=df)
# plt.show()
# df.plot(kind='line')
# plt.show()
# plt.plot(df.index, df.income)
# plt.show()
# df.income.plot()
# plt.show()
# df['income'].plot()
# plt.show()
# sns.lineplot(x=df.index, y=df.income)
# plt.show()

# 1.7.2
# df = pd.read_csv('dataset_209770_6.txt', sep=" ")
# df.plot.scatter(x = 'x', y = 'y')
# plt.show()

# 1.7.3
# df = pd.read_csv('https://stepik.org/media/attachments/course/4852/genome_matrix.csv', index_col=0)
# g = sns.heatmap(df, cmap="viridis")
# g.xaxis.set_ticks_position('top')
# g.xaxis.set_tick_params(rotation=90)
# plt.show()

# 1.7.4
# df = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')
# df['counts'] = df.roles.str.count(',')+1
# df.counts.hist()
# plt.show()

# 1.7.5
# df = pd.read_csv('https://stepik.org/media/attachments/course/4852/iris.csv')
# sns.distplot(df['petal width'], color = "blue")
# sns.distplot(df['petal length'], color ="green")
# sns.distplot(df['sepal width'], color = "yellow")
# sns.distplot(df['sepal length'], color = "orange")
# plt.show()

# 1.7.6
# df = pd.read_csv('https://stepik.org/media/attachments/course/4852/iris.csv')
# sns.violinplot(y = df['petal length'])
# plt.show()

# 1.7.7
# df = pd.read_csv('https://stepik.org/media/attachments/course/4852/iris.csv')
# sns.pairplot(df, hue = 'species')
# plt.show()

# 1.8.1
# my_data = pd.DataFrame(data={'type':['A','A','B','B'],'value':[10,14,12,23]})

# 1.8.2
# subset_1 = my_stat.head(10).iloc[:,[0,2]]
# subset_2 = my_stat.iloc[:,[1,3]].drop(my_stat.index[[0,4]])

# 1.8.3
# subset_1 = my_stat[(my_stat['V1'] > 0) & (my_stat['V3'] == 'A')]
# subset_2 = my_stat[(my_stat['V2'] != 10) | (my_stat['V4'] >= 1)]

# 1.8.4
# my_stat['V5'] = my_stat.V1 + my_stat.V4
# my_stat['V6'] = np.log(my_stat.V2)

# 1.8.5
# my_stat = my_stat.rename(columns={'V1' : 'session_value', 'V2' : 'group','V3' : 'time','V4' : 'n_users'})

# 1.8.6
# my_stat = my_stat.fillna(0)
# med = my_stat[my_stat['n_users'] >= 0].n_users.median()
# my_stat = my_stat.replace(to_replace={'n_users': my_stat.n_users[my_stat.n_users < 0]}, value={'n_users': med})

# 1.8.7
# mean_session_value_data = my_stat.groupby('group', as_index=False).agg({'session_value': 'mean'})\
#     .rename(columns = {'session_value' : 'mean_session_value'})

# 1.11.1
# events = pd.read_csv('event_data_train.csv')
# submissions = pd.read_csv('submissions_data_train.csv')
# events['date'] = pd.to_datetime(events.timestamp, unit = 's')
# events['day'] = events.date.dt.date
# submissions['date'] = pd.to_datetime(submissions.timestamp, unit = 's')
# five = events.groupby('user_id', as_index=False).count().sort_values('action', ascending= False).head(5).user_id.values
# for i in five:
#     events[events['user_id'] == i].groupby('day').agg({'action': 'count'}).plot(title = i)
#     plt.show()