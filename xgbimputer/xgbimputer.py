# Author: Leonardo de Paula Liebscher
# License: Apache-2.0

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import as_float_array

from xgboost import XGBClassifier, XGBRegressor

from .utils import encode_categories, get_inferred_categories_index, replace_categorical_values_back

__all__ = ['XGBImputer']

class XGBImputer(BaseEstimator, TransformerMixin):

    '''
    XGBImputer is an effort to implement the concepts of the MissForest
    algorithm proposed by Daniel J. Stekhoven and Peter Bühlmann in 2012,
    but leveraging the robustness and predictive power of the XGBoost
    algorithm released in 2014. It also aims to simplify the process of
    imputing categorical values in a scikit-learn compatible way.

    Parameters
    ----------

    categorical_features_index : List[int], np.array[int], optional (default = [])
        List or array of integers representing the index of categorical
        features of the array being imputed.If no index of categorical 
        feature is informed, the algorithm will treat all features as numerical.
    
    replace_categorical_values_back : bool, optional (default = False)
        If set to True, the values of the imputed X will be replaced with
        the initial categories back again. Otherwise, the categorical features
        will be OrdinalEncoded and the imputed X will be a numpy array
        containing only floats.
    
    **kwargs:
        Any keyword argument provided will be treated as XGBoost parameters
        to be set.

    Attributes
    ----------

    encoded_categories : dict
        Dictionary containing all column indexes of categorical features
        and the respective OrdinalEncoded values used by the algorithm.
    
    casted_as_string_categories : dict
        Dictionary containing all column indexes of categorical featues
        that needed to be casted as string to be processed.
        
        By default, the sklearn's OrdinalEncoder cannot operate on arrays that
        contains both floats and strings. Verifying this condition, XGBImputer
        will treat all values of the specific feature as strings to be OrdinalEncoded.

        If the parameter 'replace_categorical_values_back' is set to True, the feature's
        dtype will be casted again to 'object' and the numeric like values will be casted
        as floats when possible.
    '''
    
    def __init__(
        self,
        categorical_features_index=[],
        replace_categorical_values_back=False,
        **kwargs
    ):
        
        self.categorical_features_index = np.array(categorical_features_index)
        self.replace_categorical_values_back = replace_categorical_values_back
        if kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = {}
    
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):

        if type(X) != np.ndarray:
            X = np.array(X)

        self.columns_index = np.arange(X.shape[1])
        X, self.encoded_categories, self.casted_as_string_categories = encode_categories(X, self.categorical_features_index)

        X = as_float_array(X)
        self.isnan_array = np.isnan(X)
        
        self.inferred_features_index = get_inferred_categories_index(self)
        self.inferred_categorical_features_index = np.intersect1d(self.inferred_features_index, self.categorical_features_index)
        self.numerical_features_index = np.setdiff1d(self.columns_index, self.categorical_features_index)
        self.inferred_numerical_features_index = np.intersect1d(self.inferred_features_index, self.numerical_features_index)
        
        mean_simple_imputer = SimpleImputer(strategy='mean')
        X[:,self.numerical_features_index] = mean_simple_imputer.fit_transform(X[:,self.numerical_features_index])
        
        mode_simple_imputer = SimpleImputer(strategy='most_frequent')
        X[:,self.categorical_features_index] = mode_simple_imputer.fit_transform(X[:,self.categorical_features_index])
        
        Ximp = X.copy()
        
        iterations_counter = 1
        gamma_inferred_categorical_features_old = np.inf
        gamma_inferred_categorical_features_new = 0
        gamma_inferred_numerical_features_old = np.inf
        gamma_inferred_numerical_features_new = 0

        while (gamma_inferred_categorical_features_new < gamma_inferred_categorical_features_old or gamma_inferred_numerical_features_new < gamma_inferred_numerical_features_old):

            Ximp_old = Ximp.copy()

            if iterations_counter > 1:
                gamma_inferred_categorical_features_old = gamma_inferred_categorical_features_new
                gamma_inferred_numerical_features_old = gamma_inferred_numerical_features_new

            for column_index in self.inferred_features_index:

                if column_index in self.categorical_features_index:
                    xgb = XGBClassifier(subsample=0.7, use_label_encoder=False, verbosity=0)
                else:
                    xgb = XGBRegressor(subsample=0.7, verbosity=0)
                
                if self.kwargs:
                    xgb.set_params(**self.kwargs)

                X_obs = np.delete(Ximp, column_index, axis=1)[np.invert(self.isnan_array[:,column_index]),:]
                y_obs = Ximp[np.invert(self.isnan_array[:,column_index]),column_index]
                X_mis = np.delete(Ximp, column_index, axis=1)[self.isnan_array[:,column_index],:]
                
                one_hot_encoded_features_index = np.hstack([self.categorical_features_index[self.categorical_features_index < column_index],self.categorical_features_index[self.categorical_features_index > column_index]-1])
                ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), one_hot_encoded_features_index)])

                X_obs = ct.fit_transform(X_obs)
                X_mis = ct.transform(X_mis)

                xgb.fit(X_obs, y_obs)

                y_mis = xgb.predict(X_mis)

                Ximp[self.isnan_array[:,column_index],column_index] = y_mis

            gamma_inferred_categorical_features_new = np.sum(Ximp[:,self.inferred_categorical_features_index] != Ximp_old[:,self.inferred_categorical_features_index])/self.inferred_categorical_features_index.size
            gamma_inferred_numerical_features_new = np.sum((Ximp[:,self.inferred_numerical_features_index] - Ximp_old[:,self.inferred_numerical_features_index])**2)/np.sum((Ximp[:, self.inferred_numerical_features_index]) ** 2)
            
            print(f'XGBImputer - Epoch: {iterations_counter} | Categorical gamma: {np.format_float_positional(gamma_inferred_categorical_features_old, precision=4)}/{np.format_float_positional(gamma_inferred_categorical_features_new, precision=4)} | Numerical gamma: {np.format_float_positional(gamma_inferred_numerical_features_old, precision=10)}/{np.format_float_positional(gamma_inferred_numerical_features_new, precision=10)}')
            
            iterations_counter += 1
            
        if self.replace_categorical_values_back:
            Ximp = replace_categorical_values_back(Ximp, self.encoded_categories, self.casted_as_string_categories)
        return Ximp
    
    def fit_transform(self, X, y=None):
    
        return self.fit(X).transform(X)