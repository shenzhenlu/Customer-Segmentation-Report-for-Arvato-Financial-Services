    #!/usr/bin/env python
    # coding: utf-8

import numpy as np
import pandas as pd

def features_with_description_cleaning(features, feature_NaN_indicators, features_type):

    # =============================================================================
    # Change NaN indicator of 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015', 'CAMEO_INTL_2015' 
    # =============================================================================
    # CAMEO_DEUG_2015
    features.CAMEO_DEUG_2015 = features.CAMEO_DEUG_2015.replace(['X', np.nan], -1).astype('int')
    features.CAMEO_DEU_2015.replace(['XX', np.nan], -1, inplace=True)
    features.CAMEO_INTL_2015 = features.CAMEO_INTL_2015.replace(['XX', np.nan], -1).astype('int')
    
    # =============================================================================
    # Formalizing Missing values 
    # =============================================================================
	# Get NaN indicators of corresponding feature and store in a list
    NaN_indicators_list = []
    for NaN_indicator_str in feature_NaN_indicators:
        NaN_indicators_list.append(list(map(int, NaN_indicator_str.split(','))))
    
    # Replace NaN indicators with NaNs
    for i, NaN_indicator in enumerate(NaN_indicators_list):
        features.loc[:, features.columns[i]].replace(NaN_indicator, np.nan, inplace=True)

    # =============================================================================
    # Data Type Concatenation and Feature Engineering
    # =============================================================================
    # Concatenate mixed and categorical data
    categorical_feature_names = features_type.index[(features_type=='mixed') | (features_type=='categorical')].to_list()
    categorical_features = features[categorical_feature_names]
    
    # Select discrete data and feature engineering
    discrete_feature_names = features_type.index[features_type=='discrete'].to_list()
    discrete_features = features[discrete_feature_names]
    
    outlier_upper, outlier_lower = discrete_features.mean() + 3*discrete_features.std(), discrete_features.mean() - 3*discrete_features.std()
    discrete_features[(discrete_features > outlier_upper) | (discrete_features < outlier_lower)] = np.nan
    discrete_features.ANZ_HAUSHALTE_AKTIV = np.log(discrete_features.ANZ_HAUSHALTE_AKTIV)

    # Concatenate discrete and ordinal data
    ordinal_feature_names = features_type.index[features_type=='ordinal'].to_list()
    ordinal_features = features[ordinal_feature_names]
    
    numerical_feature_names = np.array(discrete_feature_names + ordinal_feature_names)
    numerical_features = pd.concat([discrete_features, ordinal_features], axis=1)
    
    # Combine all data together
    features_clean = pd.concat([categorical_features, numerical_features], axis=1)
    


    return features_clean, categorical_feature_names, numerical_feature_names


