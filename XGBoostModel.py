import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from GetAccountData import AccountData
import matplotlib.pyplot as plt

"""
Random Forest training model is implemented here. The data is gathered from GetAccountData.
"""

class XGBoostEvaluation:
    def xgBoostModel(self):
        features, features_high, features_low  = AccountData.getAccountData()

        #print(features)
        # Labels are the values we want to predict
        labels = np.array(features['Rating_id'])
        # Remove the labels from the features
        # axis 1 refers to the columns
        features = features.drop('Rating_id', axis=1)
        features = features.drop('ID', axis=1)
        features = features.drop('Reviews', axis=1)
        # Saving feature names for later use
        feature_list = list(features.columns)
        # Convert to numpy array
        features = np.array(features)

        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.3,
                                                                  random_state=42)

        param = {
            'eta': 0.3,
            'max_depth': 3,
            'objective': 'multi:softprob',
            'num_class': 3}

        steps = 20
        model = XGBClassifier()
        model.fit(X_train, Y_train)
        from sklearn.metrics import precision_score, recall_score, accuracy_score

        predictions = model.predict(X_test)
        #print(predictions)
        # Calculate the absolute errors
        errors = abs(predictions - Y_test)
        #print(errors)

        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2))

        # Calculate mean absolute percentage error (MAPE)
        mape = 100 - round(np.mean(errors), 2)
        print("Accuracy:", mape, '%.')

        # Get numerical feature importances
        importances = list(model.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        # Print out the feature and importances
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

        from sklearn.metrics import recall_score, precision_score
        from sklearn.metrics import precision_recall_fscore_support as score

        recall = recall_score(Y_test, predictions, average='weighted')
        precision = precision_score(Y_test, predictions, average='weighted')

        print('\nPrecision:', precision)

        print('Recall:', recall)

        return feature_list

