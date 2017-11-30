import pandas as pd
from sklearn.model_selection import train_test_split

# read data
df = pd.read_csv('historical_data.csv')

# train/test split with ratio 0.7/0.3
X_train, X_test = train_test_split(df, test_size=0.3)

# output to csv
X_train.to_csv('train.csv', index=False)
X_test.to_csv('test.csv', index=False)
