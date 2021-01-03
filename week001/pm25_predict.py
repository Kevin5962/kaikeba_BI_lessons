#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 打印出影响pm25指标的8个特征的时序数据图
from pandas import read_csv
import matplotlib.pyplot as plt

dataset = read_csv('./pollution.csv',index_col=0)
values = dataset.values
print(dataset.head())


# In[10]:


# 将分类特征wnd_dir进行标签编码(风向如NE等转换为float32类型)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# print(values[:,4])
# 设置数据类型为float32
values = values.astype('float32')
# 数据归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)
scaled.shape 
# ValueError: bad input shape (43800, 8)


# In[5]:


from pandas import DataFrame
from pandas import concat
# 监督学习，就是带有Label标签（分类，回归任务）
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    功能：
    将时间序列数据转换为适用于监督学习的数据集
    入参:
        data: 观察序列，一般为 list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: 是否删除带有NaN的行.
    出参:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除NaN行
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 将时间序列数据转换为适合监督学习的数据
reframed = series_to_supervised(scaled, 1, 1)
reframed.to_csv('reframed-1.csv')

# 去掉不需要预测的列，即 var2(t) var3(t) var4(t) var5(t) var6(t) var7(t) var8(t)
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
reframed.to_csv('reframed-2.csv')
reframed


# In[20]:


# 将原始数据以二八比例分为测试集和训练集
values = reframed.values
# lSTM不能采用train_test_split()，因为它会导致时间不连续
n_train_hours = int(len(values)*0.8)
# : n_train_hours代表前n_trains_hours行，：表示所有列
train = values[: n_train_hours,:]
# n_train_hours : 代表n_trains_hours行到结束行，：表示所有列
test = values[n_train_hours :,:]
# :-1代表从0到数组最后一位，-1代表数组最后一位
train_x = train[:,: -1];train_y = train[:,-1]
test_x = test[:,: -1];test_y = test[:,-1]
train_x.shape


# In[21]:


# 转换数据为3D格式，【样本数】【时间步】【特征数】
train_x =  train_x.reshape(train_x.shape[0],1,train_x.shape[1])
test_x =  test_x.reshape(test_x.shape[0],1,test_x.shape[1])


# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#设置网络模型
model = Sequential()
model.add(LSTM(50,input_shape=(train_x.shape[1],train_x.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#模型训练
result = model.fit(train_x,train_y,epochs=10,batch_size=64,validation_data=(test_x,test_y),verbose=2,shuffle=False)


# In[26]:


# 模型预测
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)


# In[30]:


# 绘制训练损失和测试损失
line1 = result.history['loss']
line2 = result.history['val_loss']
plt.plot(line1,label='train',c='g')
plt.plot(line2,label='test',c='r')
# 图例 train test
plt.legend(loc='best')  
plt.show


# In[32]:


model.summary()


# In[34]:


# 呈现  原始数据 训练结果 预测结果
def plot_img(source_data_set,train_predict,test_predict):
    # 原始数据 蓝色 
    plt.plot(source_data_set[:,-1],label='real',c='b')
    # 训练数据 绿色 
    plt.plot([x for x in train_predict],label='train_predict',c='g')
    # 预测数据 红色 
    plt.plot([None for _ in train_predict]+[x for x in test_predict],label='test_predict',c='r')
    # 图例
    plt.legend(loc='best')
    plt.show()
    
# 绘制预测结果与实际结果的对比
plot_img(values,train_predict,test_predict)


# In[ ]:




