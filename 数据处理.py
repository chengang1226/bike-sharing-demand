#!/usr/bin/env python
# -*- coding : utf-8 -*-
# @Time : 2019/6/12 15:28
# @Author : 陈港
# @File : 数据处理.py
# @Software : PyCharm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# 导入并查看训练数据和测试数据
train_data = pd.read_csv('./data/train.csv', header=0)
test_data = pd.read_csv('./data/test.csv', header=0)
print(train_data.shape)
print(train_data.info())
print(test_data.shape)
print(test_data.info())

# 数据预处理
# 合并两种数据，使之共同进行数据规范化
data = train_data.append(test_data)
# 拆分年、月、日、时
data['year'] = data.datetime.apply(lambda x: x.split()[0].split('-')[0])
data['year'] = data['year'].apply(lambda x: int(x))
data['month'] = data.datetime.apply(lambda x: x.split()[0].split('-')[1])
data['month'] = data['month'].apply(lambda x: int(x))
data['day'] = data.datetime.apply(lambda x: x.split()[0].split('-')[2])
data['day'] = data['day'].apply(lambda x: int(x))
data['hour'] = data.datetime.apply(lambda x: x.split()[1].split(':')[0])
data['hour'] = data['hour'].apply(lambda x: int(x))
data['date'] = data.datetime.apply(lambda x: x.split()[0])
data['weekday'] = pd.to_datetime(data['date']).dt.weekday_name
data['weekday'] = data['weekday'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                                       'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
data = data.drop('datetime', axis=1)
# 重新安排整体数据的特征
cols = ['year', 'month', 'day', 'weekday', 'hour', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
        'humidity', 'windspeed', 'casual', 'registered', 'count']
data = data.ix[:, cols]
# 分离训练数据与测试数据
train = data.iloc[:10886]
test = data.iloc[10886:]

# 观察一些重要特征分布
fig, axes = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.3, hspace=0.5)
sn.distplot(data['temp'], ax=axes[0, 0])
sn.distplot(data['atemp'], ax=axes[0, 1])
sn.distplot(data['humidity'], ax=axes[1, 0])
sn.distplot(data['windspeed'], ax=axes[1, 1])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
axes[0, 0].set(xlabel='temp', title='气温分布')
axes[0, 1].set(xlabel='atemp', title='体感温度分布')
axes[1, 0].set(xlabel='humidity', title='湿度分布')
axes[1, 1].set(xlabel='windspeed', title='风速分布')
plt.savefig('分布分析.png')

# 用随机森林预测风速
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
speed_null = data[data['windspeed'] == 0]
speed_notnull = data[data['windspeed'] != 0]
windspeed_trainX = speed_notnull[['season', 'weather', 'humidity', 'month', 'temp', 'year', 'atemp']]
windspeed_trainY = speed_notnull['windspeed']
windspeed_testX = speed_null[['season', 'weather', 'humidity', 'month', 'temp', 'year', 'atemp']]
rf = RandomForestRegressor(random_state=10)

param1 = {'n_estimators': [100, 500, 50]}
model1 = GridSearchCV(estimator=rf, param_grid=param1, scoring='neg_mean_squared_error', cv=5)
model1.fit(windspeed_trainX, windspeed_trainY)
model1.best = model1.best_params_
print('model1 best param:', model1.best_params_)
print('model1 best score:', model1.best_score_)
param2 = {'max_depth': [5, 10, 15], 'min_samples_split': [10, 5, 2]}
model2 = GridSearchCV(estimator=RandomForestRegressor(random_state=10, n_estimators=450), param_grid=param2,
                      scoring='neg_mean_squared_error', cv=5)
model2.fit(windspeed_trainX, windspeed_trainY)
model2.best = model2.best_params_
print('model2 best param:', model2.best_params_)
print('model2 best score:', model2.best_score_)
# 选择最优参数进行预测
speed_model = RandomForestRegressor(n_estimators=450, random_state=10, max_depth=10, min_samples_split=5)
speed_model.fit(windspeed_trainX, windspeed_trainY)
windspeed_testY = speed_model.predict(windspeed_testX)
data.loc[data.windspeed == 0, 'windspeed'] = windspeed_testY

# 填充后的数据特征分布
fig, axes = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.3, hspace=0.5)
sn.distplot(data['temp'], ax=axes[0, 0])
sn.distplot(data['atemp'], ax=axes[0, 1])
sn.distplot(data['humidity'], ax=axes[1, 0])
sn.distplot(data['windspeed'], ax=axes[1, 1])
axes[0, 0].set(xlabel='temp', title='气温分布')
axes[0, 1].set(xlabel='atemp', title='体感温度分布')
axes[1, 0].set(xlabel='humidity', title='湿度分布')
axes[1, 1].set(xlabel='windspeed', title='风速分布')
plt.savefig('修正后分布分析.png')

# 计算相关系数
correlation = train.corr()
influence_order = correlation['count'].sort_values(ascending=False)
influence_order_abs = abs(correlation['count']).sort_values(ascending=False)
print(influence_order)
print(influence_order_abs)

# 作相关性分析的热力图
f, ax = plt.subplots(figsize=(16, 16))
cmap = sn.cubehelix_palette(light=1, as_cmap=True)
sn.heatmap(correlation, center=1, annot=True, cmap=cmap, linewidths=2, ax=ax)
plt.show()
plt.savefig('相关性分析.png')

# 每个特征对租赁量的影响
# 时间对租赁量的影响
# (1)时间维度——年份
sn.boxplot(train['year'], train['count'])
plt.title("The influence of year")
plt.show()
plt.savefig('年份对租赁总数的影响.png')
# (2)时间维度——月份
sn.pointplot(train['month'], train['count'])
plt.title("The influence of month")
plt.show()
plt.savefig('月份对租赁总数的影响.png')
# (3)时间维度——季节
sn.boxplot(train['season'], train['count'])
plt.title("The influence of season")
plt.show()
plt.savefig('季节对租赁总数的影响.png')
# (4)时间维度——时间（小时）
sn.barplot(train['hour'], train['count'])
plt.title("The influence of hour")
plt.show()
plt.savefig('小时对租赁总数的影响.png')

# 工作日和节假日的影响
fig, axes = plt.subplots(2, 1)
plt.subplots_adjust(hspace=0.5)
ax1 = plt.subplot(2, 1, 1)
sn.pointplot(train['hour'], train['count'], hue=train['workingday'], ax=ax1)
ax1.set_title("The influence of hour (workingday)")
ax2 = plt.subplot(2, 1, 2)
sn.pointplot(train['hour'], train['count'], hue=train['holiday'], ax=ax2)
ax2.set_title("The influence of hour (holiday)")
plt.show()
plt.savefig('工作日和节假日下小时对租赁总数的影响.png')

# 天气的影响
sn.boxplot(train['weather'], train['count'])
plt.title("The influence of weather")
plt.show()
plt.savefig('天气对租赁总数的影响.png')

# 温度、湿度、风速的影响
cols = ['temp', 'atemp', 'humidity', 'windspeed', 'count']
sn.pairplot(train[cols])
plt.show()
plt.savefig('温度、湿度、风速与租赁总数的相关关系图.png')
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)
sn.regplot(train['temp'], train['count'], ax=ax1)
sn.regplot(train['humidity'], train['count'], ax=ax2)
sn.regplot(train['windspeed'], train['count'], ax=ax3)
ax1.set_title("The influence of temperature")
ax2.set_title("The influence of humidity")
ax3.set_title("The influence of windspeed")
plt.show()
plt.savefig('温度、湿度、风速对租赁总数的影响.png')


