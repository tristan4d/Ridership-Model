import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

# Read in data and perform some clean up

df = pd.read_csv('../../../Data/all_data.csv')
correlation = pd.DataFrame(df.drop(['start_of_week'],axis=1).corr()['total_boardings'])
model_data = df[correlation.dropna().index.values].drop(['local_business_condition_index'], axis=1)
features = model_data.drop(['total_boardings'], axis=1)
features['runtime_hours'] = features['runtime_hours'].rolling(window=10,min_periods=1,center=True).mean()
features = features.fillna(method='backfill')

cor_matrix = model_data.corr().abs()
to_drop = []
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
for i, j in upper_tri.iterrows():
    for x in range(len(j)):
        if j[x] > 0.9:
            if cor_matrix[i]['total_boardings'] > cor_matrix[j.index[x]]['total_boardings']:
                if j.index[x] not in to_drop:
                    to_drop.append(j.index[x])
            else:
                if i not in to_drop:
                    to_drop.append(i)

features = features.drop(to_drop,axis=1)

# Scale the data for model training.

X = features
y = model_data['total_boardings']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Deep fully-connected neural network.
# Rectified linear unit activation functions for non-ouput layers.
# Adam optimizer and mse loss function for regression on a single output variable.

model = Sequential()

model.add(Dense(21,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='Adam', loss='mse')

# Fit the model to all our data - 60 epochs chosen based on test runs with validation data.

model.fit(x=X,y=y,batch_size=1,epochs=60)

# model.save('nn_model-v2')

# Plot predicted vs. true values

predictions = model.predict(X)
x = features.index.values
plt.figure(figsize=(12,8))
plt.plot(x,y,'r-',x,predictions,'b-')

plt.show()