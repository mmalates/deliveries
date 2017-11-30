import sys
import pickle
import pandas as pd
import numpy as np
import dill


class Deliveries(object):

    def __init__(self, filename=None, pkl_file=None, dummy_columns=None):
        # read data to predict
        try:
            in_file = sys.argv[1]
        except IndexError:
            in_file = None
        if (in_file == None) & (filename == None):
            return "Provide input data, please"
        if in_file != '-f':
            self.X = pd.read_json(in_file, lines=True, convert_axes=True)
        else:
            self.X = pd.read_json(filename, lines=True, convert_axes=True)

        # load prediction model
        with open(pkl_file, 'rb') as pkl:
            self.model = dill.load(pkl)

        # read in features and build feature list
        self.features_ = ['delivery_id']
        with open('features.txt', 'r') as f:
            for line in f:
                self.features_.append(line.split('\n')[0])

        self.dummy_columns = dummy_columns

    def processing(self, X):
        '''
        Processes and cleans dataframe.  Updates dataframe with dummy columns from self.dummy_columns.

        Parameters
        ----------------------
        X : DataFrame to process

        Returns
        ----------------------
        Processed DataFrame
        '''
        # convert numeric columns from object to numeric types
        numeric_columns = ['estimated_store_to_consumer_driving_duration',
                           'market_id', 'order_protocol', 'total_busy_runners',
                           'total_onshift_runners', 'total_outstanding_orders']
        for col in numeric_columns:
            X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')

        # shift timezone
        X.loc[:, 'created_at'] = pd.to_datetime(X.loc[:, 'created_at']).dt.tz_localize(
            'utc').dt.tz_convert('US/Pacific')

        # Make a total items cutoff.  Drop great total items to the max value (12 items)
        mask_total_items = X.total_items > 12
        X.loc[mask_total_items, 'total_items'] = 12

        # creat hour of day feature
        X.loc[:, 'hour'] = X.loc[:, 'created_at'].dt.hour

        # create lunch and dinner features
        X.loc[:, 'breakfast'] = (
            X.hour >= 6) & (X.hour <= 8)
        X.loc[:, 'lunch'] = (X.hour >= 11) & (
            X.hour <= 13)
        X.loc[:, 'dinner'] = (X.hour >= 17) & (
            X.hour <= 19)

        # create day of week feature
        X.loc[:, 'DOW'] = X.loc[:, 'created_at'].dt.dayofweek

        # # merge store times for fast_store, slow_store, and mean_store_time features
        store_times = pd.read_csv('store_times.csv')
        X = X.merge(store_times, how='left', on='store_id')

        # create busy runners: total runners ratio feature
        X.loc[:, 'fraction_busy_runners'] = X.loc[:,
                                                  'total_busy_runners'] / X.loc[:, 'total_onshift_runners']
        X.loc[X['total_onshift_runners'].isnull(), 'total_onshift_runners'] = 0
        X.loc[X['store_primary_category'].isnull(
        ), 'store_primary_category'] = 'None'

        # fix infinities and NaNs
        X = X.replace([np.inf, -np.inf], 0.0)
        X = X.fillna(X.median())

        # drop columns that won't be used
        X = X.drop(['created_at', 'store_id', 'total_busy_runners'], axis=1)

        # dummify columns and make congruent with pickle features
        if self.dummy_columns != None:
            X = self._dummify_categories(X, self.dummy_columns)
        for column in X.columns:
            if column not in self.features_:
                X = X.drop(column, axis=1)
        for column in self.features_:
            if column not in X.columns:
                X.loc[:, column] = 0

        return X

    def _dummify_categories(self, X, to_dummy_cols):
        '''
        Parameters
        ----------------------
        to_dummy_cols : string name of column

        Returns
        ----------------------
        Dataframe with column replaced with dummy columns
        '''
        for column in to_dummy_cols:
            X_dummies = pd.get_dummies(X[column], prefix=column)
            X = pd.concat((X, X_dummies), axis=1)
            X = X.drop(column, axis=1)
        return X

    def predict(self):
        '''
        Predicts delivery times for data in self.X using self.model
        '''
        self.X['predictions'] = pd.Series(
            np.array(self.model.predict(self.X.drop('delivery_id', axis=1))))
        self.X[['delivery_id', 'predictions']].to_csv(
            'predictions.csv', sep='\t', index=False)


if __name__ == '__main__':
    # specify dummy columns
    dummy_columns = ['DOW', 'hour', 'market_id',
                     'order_protocol', 'store_primary_category']

    # instanciate Deliveries class object
    deliveries = Deliveries(filename='data_to_predict.json',
                            pkl_file='rf.dill', dummy_columns=dummy_columns)

    # process data
    deliveries.X = deliveries.processing(deliveries.X)

    # predict delivery times and write to file
    deliveries.predict()
    print 'Predicted delivery times written to predictions.csv'
