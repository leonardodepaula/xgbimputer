
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def is_float(value):
    '''
    Verify if value's dtype is float.
    '''
    try:
        float(value)
    except ValueError:
        return False
    else:
        return True

is_float_vectorized = np.vectorize(is_float)

def is_nan(value):
    '''
    Verify if a value is NaN.
    '''
    return value != value

is_nan_vectorized = np.vectorize(is_nan)

def is_string(value):
    '''
    Verify if value's dtype is string.
    '''
    return isinstance(value, str)

is_string_vectorized = np.vectorize(is_string)

def is_not_string(value):
    '''
    Verify if value's dtype isn't string.
    '''
    if not isinstance(value, str) and value == value:
        return True
    else:
        return False
    
is_not_string_vectorized = np.vectorize(is_not_string)

def encode_categories(X, categorical_features_index):
    '''
    Given an X and an array of categorical features for X,
    the function iterates over all categorical features to
    get original and new values of OrdinalEncoded features.

    It also gets categorical features that needed to be casted
    as string for containing both float and string dtypes.
    '''
    encoded_categories = {}
    casted_as_string_categories = []
    for column_index in categorical_features_index:
        ordinal_encoder = OrdinalEncoder()
        if np.any(is_string_vectorized(X[:,column_index])) and np.any(is_not_string_vectorized(X[:,column_index])):
            X[:,column_index] = ordinal_encoder.fit_transform(X[np.invert(is_nan_vectorized(X[:,column_index])),column_index].reshape(-1,1).astype(str)).reshape(1,-1)
            casted_as_string_categories.append(column_index)
        else:
            X[:,column_index] = ordinal_encoder.fit_transform(X[:,column_index].reshape(-1,1)).reshape(1,-1)
        context_categories = [value for value in ordinal_encoder.categories_[0] if not is_nan(value)]
        encoded_categories[column_index] = dict(zip(range(len(context_categories)), context_categories))
    return X, encoded_categories, casted_as_string_categories

def get_inferred_categories_index(self):
    '''
    Get an array containing the index of the categorical features
    that were inferred by XGBImputer.
    '''
    inferred_features_index = np.hstack([np.nonzero(self.isnan_array.sum(axis=0))[0].reshape(-1,1), self.isnan_array.sum(axis=0)[self.isnan_array.sum(axis=0) > 0].reshape(-1,1)])
    inferred_features_index = inferred_features_index[inferred_features_index[:, 1].argsort()]
    return inferred_features_index[:,0]

def replace_categorical_values_back(X, encoded_categories, casted_as_string_categories):
    '''
    Replace the values of the imputed X with the initial categories back again.
    '''
    X = X.astype('object')
    for key, values in encoded_categories.items():
        replace_func = np.vectorize(values.get)
        X[:,key] = replace_func(X[:,key])
        if key in casted_as_string_categories:
            X[:,key] = X[:,key].astype('object')
            X[is_float_vectorized(X[:,key]),key] = X[is_float_vectorized(X[:,key]),key].astype(np.float64)
    return X