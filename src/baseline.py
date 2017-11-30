import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as skm
import pandas as pd

# read data
df = pd.read_csv('../data/train.csv')

# perform processing
df.loc[:, 'created_at'] = pd.to_datetime(df.loc[:, 'created_at']).dt.tz_localize(
    'utc').dt.tz_convert('US/Pacific')
df.loc[:, 'actual_delivery_time'] = pd.to_datetime(
    df.loc[:, 'actual_delivery_time']).dt.tz_localize('utc').dt.tz_convert('US/Pacific')
df.loc[:, 'delivery_time'] = df.loc[:, 'actual_delivery_time'].sub(
    df.loc[:, 'created_at']).dt.total_seconds()
mask_total_items = df.total_items > 12
df.loc[mask_total_items, 'total_items'] = 12
df = df.replace([np.inf, -np.inf], 0.0)
df = df.dropna()


X = df['total_items'].values
y = df['delivery_time'].values

model = LinearRegression()

kf = KFold(n_splits=10, shuffle=True)
error = []
for train_index, test_index in kf.split(y):
    y_train, y_test = y[train_index], y[test_index]
    X_train, X_test = X.reshape(-1,
                                1)[train_index], X.reshape(-1, 1)[test_index]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))

print 'train error: ', np.mean(error)
