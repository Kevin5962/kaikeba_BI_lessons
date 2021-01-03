#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 将原始数据格式中的year, month, day, hour进行合并，并保存新的文件preprocess_data.csv
from pandas import read_csv
from datetime import datetime
# 合并数据集中的所有时间单位
def parse(x):
    return datetime.strptime(x,'%Y %m %d %H')
# 数据加载
# x =  read_csv('./raw.csv')
dataset = read_csv('./raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
# 打印表头
print(dataset.head())
# 删除行号
dataset.drop('No', axis=1, inplace=True)
# 列名替换
dataset.columns = ['pm25', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# 缺失值填充
dataset['pm25'].fillna(0, inplace=True)
# 去掉第一天数据（前24小时）
dataset = dataset[24:]
# 数据浏览
print(dataset.head(5))
# 数据保存
dataset.to_csv('pollution.csv')


# In[ ]:




