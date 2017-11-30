import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 100)
# read training data
df = pd.read_csv('~/doordash/data/train.csv')

# subtract created_at from actual_delivery_time to get the target variable, delivery_time
df['delivery_time'] = pd.to_datetime(df['actual_delivery_time']).sub(
    pd.to_datetime(df['created_at'])).dt.total_seconds()

# convert timezone to PST
df.index = pd.to_datetime(df.created_at)
df = df.tz_localize('utc')
df = df.tz_convert('US/Pacific')
df = df.drop(['created_at', 'actual_delivery_time'], axis=1).reset_index()

# Remove unlikely delivery times
mask_delivery_time = (df.delivery_time > 500) & (df.delivery_time < 7200)
df = df[mask_delivery_time]
df.head()
# Make a total items cutoff.  Drop great total items to the max value (12 items)
mask_total_items = df.total_items > 12
df['total_items'][mask_total_items] = 12

# creat hour of day feature
df['hour'] = df['created_at'].dt.hour

# create breakfast, lunch and dinner categorical features
df['breakfast'] = (df.hour >= 6) & (df.hour <= 8)
df['lunch'] = (df.hour >= 11) & (df.hour <= 13)
df['dinner'] = (df.hour >= 17) & (df.hour <= 19)


# create day of week feature
df['DOW'] = df['created_at'].dt.dayofweek

# create month feature to capture seasonality: Later not worth it, only 2 months
df['date'] = df['created_at'].dt.strftime('%Y-%m-%d')

# get total orders for a given day
daily_counts = df.groupby(
    by='date')['created_at'].count().reset_index(name='count')


def plot_weekly_means(df, y_label):
    """plots a bar graph of y binned by day of the week"""
    dow_dict = {0: 'Su', 1: 'M', 2: 'T', 3: 'W', 4: 'Th', 5: 'F', 6: 'S'}
    df['tick_labels'] = df.DOW.map(dow_dict)
    fig, ax = plt.subplots(figsize=(8, 6))
    df['mean'].plot(kind='bar')
    ax.set_xticklabels(df.tick_labels, rotation=0)
    plt.xlabel('Day of Week')
    plt.ylabel(y_label)
    plt.title(y_label + ' vs. Day of Week')
    plt.savefig('plots/mean_' + y_label + '_vs_DOW.png')


# get the day of week for a given day
daily_counts['DOW'] = pd.to_datetime(daily_counts.date).dt.dayofweek

# group by day of week and get the mean order counts
dow_mean_orders = daily_counts.groupby(
    by='DOW')['count'].mean().reset_index(name='mean')

# group by day of week and get the mean delivery time
dow_delivery_times = df.groupby(
    by='DOW')['delivery_time'].mean().reset_index(name='mean')

plot_weekly_means(dow_mean_orders, 'Orders')
plot_weekly_means(dow_delivery_times, 'Delivery_Time')


def get_categorical_mean_delivery_time(col):
    mean_vals = df.groupby(
        by=col)['delivery_time'].mean().reset_index(name='mean')
    return mean_vals[[col, 'mean']]


df.describe().T
fig, ax = plt.subplots(6, figsize=(12, 20))
for i, col in enumerate(['market_id', 'store_id', 'store_primary_category', 'order_protocol', 'total_items', 'hour']):
    mean_df = get_categorical_mean_delivery_time(col)
    mean_df.plot(x=col, y='mean', ax=ax[i])

plt.savefig('plots/mean_delivery_vs_features.png')

store_times = get_categorical_mean_delivery_time('store_id')

sorted_stores = store_times.sort_values('mean')
sorted_stores = sorted_stores.reset_index().drop('index', axis=1)
sorted_stores
sorted_stores.plot()
sorted_stores.describe().T
plt.show()

store_times['fast_store'] = store_times['mean'] < 2491
store_times['slow_store'] = store_times['mean'] > 3228
store_times['mean_store_time'] = store_times['mean']
store_times = store_times.drop('mean', axis=1)

store_times.to_csv('../../data/store_times.csv', index=False)
df.merge(store_times[['fast_store', 'slow_store',
                      'store_id']], how='left', on='store_id')
