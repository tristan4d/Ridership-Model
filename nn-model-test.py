import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

# Read in data and perform some clean up

df = pd.read_csv('../../../Data/all_data.csv')
correlation = pd.DataFrame(df.drop(['start_of_week'],axis=1).corr()['total_boardings'])
model_data = df[correlation.dropna().index.values].drop(['local_business_condition_index','high_school_off_season'], axis=1)
features = model_data.drop(['total_boardings'], axis=1)
features['runtime_hours'] = features['runtime_hours'].rolling(window=10,min_periods=1,center=True).mean()
features = features.fillna(method='backfill').fillna(method='ffill')

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

# Perform train/test split and scale the data for model training.

X = features
y = model_data['total_boardings']

# Testing time series k-fold validation.

fh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
initial_window = 104
step_length = 26
splitter = ExpandingWindowSplitter(fh=fh,initial_window=initial_window,step_length=step_length)
splits = splitter.split(y)
rmse = []
mae = []

# Deep fully-connected neural network.
# Rectified linear unit activation functions for non-ouput layers.
# Adam optimizer and mse loss function for regression on a single output variable.

model = Sequential()

# model.add(Dense(42,activation='relu'))
model.add(Dense(21,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='Adam',loss='mse')

# for split in splits:
#     model = Sequential()

#     # model.add(Dense(42,activation='relu'))
#     model.add(Dense(21,activation='relu'))
#     model.add(Dense(10,activation='relu'))
#     model.add(Dense(5,activation='relu'))
#     model.add(Dense(1))

#     model.compile(optimizer='Adam',loss='mse')

#     X_train = X.iloc[split[0]]
#     X_test = X.iloc[split[1]]
#     y_train = y.iloc[split[0]]
#     y_test = y.iloc[split[1]]

#     scaler = MinMaxScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)

#     model.fit(x=X_train,y=y_train,validation_data=(X_test, y_test),batch_size=1,epochs=170,callbacks=[early_stop])
#     losses = pd.DataFrame(model.history.history)
#     predictions = model.predict(X_test)

#     mae.append(mean_absolute_error(y_test, predictions) / model_data['total_boardings'].mean())
#     rmse.append(np.sqrt(mean_squared_error(y_test,predictions)) / model_data['total_boardings'].mean())

#     # x = np.arange(len(X_test))
#     # plt.plot(x,y_test,'r-',x,predictions,'b-')

#     # plt.show()

# x = np.arange(len(mae))
# plt.plot(x,mae,'r-',x,rmse,'b-')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the model to the first 80% of our data and test on the last 20%.

early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)

model.fit(x=X_train,y=y_train,validation_data=(X_test, y_test),batch_size=1,epochs=170,callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
predictions = model.predict(X_test)

print('MAE: ',mean_absolute_error(y_test, predictions) / model_data['total_boardings'].mean())
print('RMSE: ',np.sqrt(mean_squared_error(y_test,predictions)) / model_data['total_boardings'].mean())

# Plot predicted vs. true values

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Total Boardings - True vs. Predicted Values')

ax1.plot(losses)

ax2.scatter(y_test,predictions)
ax2.plot(y_test,y_test,'r-',marker=None)
ax2.plot(y_test,y_test*1.1,'y-')
ax2.plot(y_test,y_test*0.9,'y-')

x = np.arange(len(X_test))
ax3.plot(x,y_test,'r-',x,predictions,'b-')

plt.show()