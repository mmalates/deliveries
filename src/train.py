import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as skm
import pickle
import dill


class Deliveries(object):

    def __init__(self, train_filename=None, test_filename=None, full_dataset_filename=None):
        '''
        Reads training and test data into dataframes and initializes attributes
        '''
        # load datasets
        if train_filename != None:
            self.X_train = pd.read_csv(train_filename)
        self.y_train = None
        if test_filename != None:
            self.X_test = pd.read_csv(test_filename)
        self.y_test = None
        if full_dataset_filename != None:
            self.X = pd.read_csv(full_dataset_filename)
        self.y = None
        # initialize data attributes
        self.features_ = None
        self.selected_features_ = []
        self.train_error_ = None
        self.test_error_ = None
        self.train_msdt = None
        self.all_msdt = None

    def _processing(self, X, y, type):
        '''
        Processes and cleans test dataframe.  Updates dataframe with new dummy columns.

        Input: dummy_columns - columns to dummify
            type : 'train', 'test', 'all'

        Output: None
        '''
        X.loc[:, 'created_at'] = pd.to_datetime(X.loc[:, 'created_at']).dt.tz_localize(
            'utc').dt.tz_convert('US/Pacific')
        X.loc[:, 'actual_delivery_time'] = pd.to_datetime(
            X.loc[:, 'actual_delivery_time']).dt.tz_localize('utc').dt.tz_convert('US/Pacific')
        X.loc[:, 'delivery_time'] = X.loc[:, 'actual_delivery_time'].sub(
            X.loc[:, 'created_at']).dt.total_seconds()

        # Cut out unlikely delivery times
        delivery_time_mask = (X['delivery_time'] < 7200) & (
            X['delivery_time'] > 300)
        X = X[delivery_time_mask]

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

        # join store times dataframe to create, fast_store, slow_store, and mean_store_times features
        # store_times = pd.read_csv('../data/store_times.csv')
        # X = X.merge(store_times, how='left', on='store_id')
        if type == 'train':
            self._training_store_mean_delivery_times()
            X = X.merge(self.train_msdt, how='left', on='store_id')
        elif type == 'test':
            X = X.merge(
                self.train_msdt, how='left', on='store_id')
        else:
            self._all_store_mean_delivery_times()
            X = X.merge(self.all_msdt, how='left', on='store_id')

        # create busy runners: total runners ratio feature
        X.loc[:, 'fraction_busy_runners'] = X.loc[:, 'total_busy_runners'] / \
            X.loc[:, 'total_onshift_runners']
        X = X.replace([np.inf, -np.inf], 0.0)
        X.loc[X['total_onshift_runners'].isnull(), 'total_onshift_runners'] = 0
        X.loc[X['store_primary_category'].isnull(
        ), 'store_primary_category'] = 'None'
        X = X.fillna(X.median())

        y = X.loc[:, 'delivery_time']
        X = X.drop(['delivery_time', 'store_id', 'created_at', 'actual_delivery_time',
                    'total_busy_runners'], axis=1)
        return X, y

    def train_processing(self, dummy_columns=None):
        '''
        Processes and cleans training dataframe.  Updates dataframe with new dummy columns.

        Input: dummy_columns - columns to dummify

        Output: None
        '''
        self.X_train, self.y_train = self._processing(
            self.X_train, self.y_train, 'train')
        if dummy_columns != None:
            self.X_train = self._dummify_categories(
                self.X_train, dummy_columns)
        self.X_train.drop('store_primary_category_None', axis=1)
        self.train_features_ = self.X_train.columns

    def test_processing(self, dummy_columns=None):
        '''
        Processes and cleans test dataframe.  Updates dataframe with new dummy columns.

        Input: dummy_columns - columns to dummify

        Output: None
        '''
        self.X_test, self.y_test = self._processing(
            self.X_test, self.y_test, 'test')
        if dummy_columns != None:
            self.X_test = self._dummify_categories(self.X_test, dummy_columns)
        for column in self.X_test.columns:
            if column not in self.X_train.columns:
                self.X_test = self.X_test.drop(column, axis=1)
        for column in self.X_train.columns:
            if column not in self.X_test.columns:
                self.X_test.loc[:, column] = 0

    def _dummify_categories(self, X, to_dummy_cols):
        '''
        Input: to_dummy_cols - string name of column

        Updates dataframe with new columns
        '''
        for column in to_dummy_cols:
            X_dummies = pd.get_dummies(X[column], prefix=column)
            X = pd.concat((X, X_dummies), axis=1)
            X = X.drop(column, axis=1)
        return X

    def train_model(self, X, y, model):
        '''
        Trains a linear model.  Don't forget to include the parameters in the model.  Updates train_error_ attribute with Kfold cross-validated training error

        Input:
            X- training data
            y- training target
            model - model class with desired parameters

        Output: trained model
        '''
        kf = KFold(n_splits=10, shuffle=True)
        error = []
        for train_index, test_index in kf.split(y):
            print 'new fold'
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))
        self.train_error_ = np.mean(error)
        return model

    def train_forest(self, X, y, model):
        '''
        Trains a linear model.  Don't forget to include the parameters in the model.  Updates train_error_ attribute with Kfold cross-validated training error

        Input:
            X- training data
            y- training target
            model - model class with desired parameters

        Output: trained model
        '''
        kf = KFold(n_splits=2, shuffle=True)
        error = []
        for train_index, test_index in kf.split(y):
            print 'new fold'
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))
        self.train_error_ = np.mean(error)
        return model

    def test_model(self, X, y, trained_model):
        '''
        Tests the performance of the trained model on test data.  Updates test_error_ attribute

        Input:
            X- test data
            y- test target to compare
            model- trained model object

        Ouput: None
        '''
        predictions = trained_model.predict(X)
        self.test_error_ = np.sqrt(skm.mean_squared_error(y, predictions))
        return predictions

    def pickle_model(self, model, filename, dummy_columns, features):
        '''
        Saves a trained model in a .pkl file

        Input:
            model- model to train
            filename-  name for the .pkl file
        '''
        # dummify columns
        print 'processing dataset'
        self.X, self.y = self._processing(self.X, self.y, 'all')
        if dummy_columns != None:
            self.X = self._dummify_categories(
                self.X, dummy_columns)

        # train model
        print 'training model'
        if self.selected_features_ != []:
            trained_model = model.fit(self.X[features], self.y)
        else:
            trained_model = model.fit(self.X, self.y)
        print str(trained_model)
        if str(trained_model)[0] == 'L':
            self._coefs_to_file(trained_model, 'coefs.txt')
        else:
            self._importances_to_file(trained_model, 'importances.txt')
        print 'pickling model'
        with open(filename, 'wb') as pkl:
            dill.dump(trained_model, pkl)

    def _coefs_to_file(self, trained_model, filename):
        '''
        Prints coefficients to a specified file

        Parameters
        ----------------------------
        train_model : trained model to get coefficients from
        filename : file to print coefficients to
        '''
        with open(filename, 'w') as f:
            for coef, feature in sorted(zip(trained_model.coef_, self.train_features_), reverse=True):
                f.write('{} : {}\n'.format(feature, coef))
                if coef not in [-0.0, 0.0]:
                    self.selected_features_.append(feature)

    def _importances_to_file(self, trained_model, filename):
        '''
        Prints coefficients to a specified file

        Parameters
        --------------------------
        train_model : trained model to get coefficients from
        filename : file to print coefficients to
        '''
        with open(filename, 'w') as f:
            for coef, feature in sorted(zip(trained_model.feature_importances_, self.train_features_), reverse=True):
                f.write('{} : {}\n'.format(feature, coef))

    def _store_mean_delivery_times(self, X):
        '''
        Calculates mean delivery times for each store

        Parameters
        -----------------------
        X : DataFrame on which to comput store means

        Returns
        -----------------------
        Updated DataFrame
        '''
        mean_vals = X.groupby(by='store_id')['delivery_time'].mean(
        ).reset_index(name='mean_store_delivery_time')
        return mean_vals[['store_id', 'mean_store_delivery_time']]

    def _training_store_mean_delivery_times(self):
        '''
        Merges mean store delivery times with DataFrame

        Parameters
        ------------------------
        X : DataFrame to merge delivery times with

        Returns
        ------------------------
        Updated DataFrame
        '''
        self.train_msdt = self._store_mean_delivery_times(self.X_train)
        self.train_msdt['fast_store'] = self.train_msdt['mean_store_delivery_time'] < 2500
        self.train_msdt['slow_store'] = self.train_msdt['mean_store_delivery_time'] > 3200

    def _all_store_mean_delivery_times(self):
        '''
        Merges mean store delivery times with DataFrame

        Parameters
        ------------------------
        X : DataFrame to merge delivery times with

        Returns
        ------------------------
        Updated DataFrame
        '''
        self.all_msdt = self._store_mean_delivery_times(self.X)
        self.all_msdt['fast_store'] = self.all_msdt['mean_store_delivery_time'] < 2500
        self.all_msdt['slow_store'] = self.all_msdt['mean_store_delivery_time'] > 3200


if __name__ == '__main__':
    # # instanciate delivery class object
    deliveries = Deliveries(
        train_filename='../data/train.csv', test_filename='../data/test.csv', full_dataset_filename='../data/historical_data.csv')
    with open('performance_lasso.txt', 'w') as lasso_file:
        # process training data
        print 'processing training data'
        dummy_columns = ['DOW', 'hour', 'market_id',
                         'order_protocol', 'store_primary_category']
        deliveries.train_processing(dummy_columns=dummy_columns)

        # tune hyper parameter with LassoCV
        print 'tuning hyper parameters'
        hyper_params_model = LassoCV(normalize=True, max_iter=2000).fit(
            deliveries.X_train, deliveries.y_train)
        lasso_file.write('best hyperparameter is: {}\n'.format(
            hyper_params_model.alpha_))
        alpha = hyper_params_model.alpha_

        # train model with cross validated hyper parameter
        print 'training model'
        lasso_model = deliveries.train_model(
            deliveries.X_train, deliveries.y_train, Lasso(alpha=alpha, normalize=True, max_iter=2000))
        lasso_file.write('model trained with error: {}\n'.format(
            deliveries.train_error_))

        # process test data and evaluate model performance
        print 'testing model'
        deliveries.test_processing(dummy_columns=dummy_columns)
        predictions = deliveries.test_model(
            deliveries.X_test, deliveries.y_test, lasso_model)
        lasso_file.write('model tested with error: {}\n'.format(
            deliveries.test_error_))

        print 'pickling model and writing coefs to file'
        deliveries.pickle_model(
            Lasso(alpha=alpha, normalize=True), '../app/lasso.dill', dummy_columns=dummy_columns, features=deliveries.selected_features_)

        with open('../app/features.txt', 'w') as feature_file:
            for col in deliveries.selected_features_:
                feature_file.write('{}\n'.format(col))
        print 'done with linear regression'

    with open('performance_tree.txt', 'w') as tree_file:
        deliveries = Deliveries(
            train_filename='../data/train.csv', test_filename='../data/test.csv', full_dataset_filename='../data/historical_data.csv')

        # process training data
        print 'processing training data'
        dummy_columns = ['DOW', 'hour', 'market_id',
                         'order_protocol', 'store_primary_category']
        with open('../app/features.txt', 'r') as feature_file:
            for line in feature_file:
                deliveries.selected_features_.append(line.split('\n')[0])
        deliveries.train_processing(dummy_columns=dummy_columns)
        print 'cross validating forest'
        forest_model = deliveries.train_forest(deliveries.X_train, deliveries.y_train, RandomForestRegressor(
            min_samples_leaf=4, max_depth=20, bootstrap=True, n_estimators=1000, n_jobs=4))
        tree_file.write('model trained with error: {}\n'.format(
            deliveries.train_error_))

        # process test data and evaluate model performance
        print 'testing model'
        deliveries.test_processing(dummy_columns=dummy_columns)
        predictions = deliveries.test_model(
            deliveries.X_test, deliveries.y_test, forest_model)
        tree_file.write('model tested with error: {}\n'.format(
            deliveries.test_error_))

        # pickle random forest model for application
        print 'pickling model'
        deliveries.pickle_model(
            RandomForestRegressor(min_samples_leaf=4, max_depth=20, bootstrap=True, n_estimators=1000, n_jobs=4), filename='../app/rf.dill', dummy_columns=dummy_columns, features=deliveries.selected_features_)
        print 'DONE'

        # print features to file
        # with open('../app/features.txt', 'w') as features:
        #     for col in deliveries.selected_features_:
        #         features.write('{}\n'.format(col))
        # deliveries.selected_features_
        deliveries.all_msdt.to_csv('../app/store_times.csv', index=False)


# with open('../app/features.txt', 'w') as feature_file:
#     for col in deliveries.X.columns:
#         feature_file.write(col)
#         feature_file.write('\n')
#
# deliveries.selected_features_
#
# selected = []
# for coef, feature in sorted(zip(lasso_model.coef_, deliveries.train_features_), reverse=True):
#     print ('{} : {}\n'.format(feature, coef))
#     if coef not in [-0.0, 0.0]:
#         print feature
#         selected.append(feature)
