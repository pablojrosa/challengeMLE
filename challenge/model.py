import pandas as pd
import os
import numpy as np
from utils import (get_period_day, is_high_season, get_min_diff,
                   shuffle_data, get_dummies_features, split_data, get_scale,
                   train_model, get_prediction, get_top_10_features)

from typing import Tuple, Union, List

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(self,
                data: pd.DataFrame,
                target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): Raw data.
            target_column (str, optional): If set, the target is returned separately.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and target, if target_column is specified.
            pd.DataFrame: Features only, if no target_column is specified.
        """

        # Features creation

        # Create period of day, high season, and minute difference features
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis=1)
        data['delay'] = np.where(data['min_diff'] > 15, 1, 0)

        data = shuffle_data(data=data)

        # Generate dummy features
        dummy_features =get_dummies_features(data)

        # Concatenate dummy features with other features
        data = pd.concat([data, dummy_features], axis=1)

        if target_column and target_column in data.columns:
            target = data[target_column]
            all_features = data.drop(columns=[target_column])
            features = get_top_10_features(all_features)
            """result = (features.to_csv('data_preprocessed_without_target.csv', index=False),
                      target.to_csv('target.csv', index=False)) # luego borrar el to_csv"""
            
            return features, target
        else:
            """result = data.to_csv('data_preprocessed.csv', index=False) # luego borrar el to_csv"""
            return data

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, x_test, y_train, y_test = split_data(features, target)
        scale = get_scale(y_train=y_train)

        trained_model = train_model(x_train, y_train, scale)
        if not os.path.exists('challenge/trained_model'):
            os.makedirs('challenge/trained_model')

        trained_model.save_model('challenge/trained_model/xgboost_model.pkl')
        
    def predict(
        self,
        features: pd.DataFrame,
        model_path: str
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        predictions = get_prediction(model_path, features)

        return predictions