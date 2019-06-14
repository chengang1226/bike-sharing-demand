#!/usr/bin/env python
# -*- coding : utf-8 -*-
# @Time : 2019/6/14 16:03
# @Author : 陈港
# @File : 模型构建.py
# @Software : PyCharm

import pandas as pd
import numpy as np
# 导入数据
train_data = pd.read_csv('./data/train.csv', header=0)
test_data = pd.read_csv('./data/test.csv', header=0)

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

# 用随机森林填充风速为0的数据
from sklearn.ensemble import RandomForestRegressor
speed_null = data[data['windspeed'] == 0]
speed_notnull = data[data['windspeed'] != 0]
windspeed_trainX = speed_notnull[['season', 'weather', 'humidity', 'month', 'temp', 'year', 'atemp']]
windspeed_trainY = speed_notnull['windspeed']
windspeed_testX = speed_null[['season', 'weather', 'humidity', 'month', 'temp', 'year', 'atemp']]

speed_model = RandomForestRegressor(n_estimators=450, random_state=10, max_depth=10, min_samples_split=5)
speed_model.fit(windspeed_trainX, windspeed_trainY)
windspeed_testY = speed_model.predict(windspeed_testX)
data.loc[data.windspeed == 0, 'windspeed'] = windspeed_testY

# 特征工程
# 所选取的特征：year、month、hour、workingday、holiday、weather、temp、humidity和windspeed
# (1) 删除不要的变量
data = data.drop(['day', 'weekday', 'season', 'atemp', 'casual', 'registered'], axis=1)
# (2) 离散型变量（year、month、hour、weather）转换
column_trans = ['year', 'month', 'hour', 'weather']
data = pd.get_dummies(data, columns=column_trans)

# 机器学习
# 1、特征向量化
col_trans = ['holiday', 'workingday', 'temp', 'humidity', 'windspeed',
             'year_2011', 'year_2012', 'month_1', 'month_2', 'month_3', 'month_4',
             'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10',
             'month_11', 'month_12', 'hour_0', 'hour_1', 'hour_2', 'hour_3',
             'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10',
             'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16',
             'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22',
             'hour_23', 'weather_1', 'weather_2', 'weather_3', 'weather_4']
X_train = data[col_trans].iloc[:10886]
X_test = data[col_trans].iloc[10886:]
Y_train = data['count'].iloc[:10886]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.fit_transform(X_test.to_dict(orient='record'))

# 分割训练数据
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=40)

# 2、建模预测，分别采用常规集成学习方法、XGBoost和神经网络三大类模型
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# （1）集成学习方法——普通随机森林
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
# print(rf.fit(x_train,y_train))
rf_y_predict = rf.predict(x_test)
print("集成学习方法——普通随机森林回归模型的R方得分为：", r2_score(y_test, rf_y_predict))

# （2）集成学习方法——极端随机森林
et = ExtraTreesRegressor(random_state=28, n_estimators=40, max_depth=38, max_features=14)
et.fit(x_train, y_train)
# print(et.fit(x_train,y_train))
et_y_predict = et.predict(x_test)
print("集成学习方法——极端随机森林回归模型的R方得分为：", r2_score(y_test, et_y_predict))

# （3）集成学习方法——梯度提升树
gb = GradientBoostingRegressor()
gb.fit(x_train, y_train)
# print(gb.fit(x_train,y_train))
gb_y_predict = gb.predict(x_test)
print("集成学习方法——梯度提升树回归模型的R方得分为：", r2_score(y_test, gb_y_predict))

# （4） XGBoost回归模型
xgb = XGBRegressor()
xgb.fit(x_train, y_train)
# print(xgb.fit(x_train,y_train))
xgb_y_predict = xgb.predict(x_test)
print("XGBoost回归模型的R方得分为：", r2_score(y_test, xgb_y_predict))

# （5） 神经网络回归模型
mlp = MLPRegressor(hidden_layer_sizes=(45, 45, 45), max_iter=500)
mlp.fit(x_train, y_train)
mlp_y_predict = mlp.predict(x_test)
print("神经网络回归模型的R方得分为：", r2_score(y_test, mlp_y_predict))

# 极端随机森林模型下的预测结果
Y_pred = et.predict(X_test)
datetest = pd.read_csv('./data/test.csv')["datetime"]
results_et = pd.DataFrame({'datetime': datetest, 'count': np.around(Y_pred)})
results_et.to_csv('./data/results_et.csv')

# 神经网络模型下的预测结果
Y_pred = mlp.predict(X_test)
datetest = pd.read_csv('./data/test.csv')["datetime"]
results_mlp = pd.DataFrame({'datetime': datetest, 'count': np.around(Y_pred)})
results_mlp.to_csv('./data/results_mlp.csv')

