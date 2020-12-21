import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from GetAccountData import AccountData
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, plot_roc_curve, f1_score


"""
Random Forest training model is implemented here. The data is gathered from GetAccountData.
"""

class XGBoostEvaluationKFold:
    def xgBoostModelKFold(self):
        features, features_high, features_low  = AccountData.getAccountData()

        #print(features)
        # Labels are the values we want to predict
        labels = np.array(features['Rating_id'])
        # Remove the labels from the features
        # axis 1 refers to the columns
        y = features['Rating_id']
        features = features.drop('Rating_id', axis=1)
        features = features.drop('ID', axis=1)
        features = features.drop('Reviews', axis=1)
        # Saving feature names for later use
        feature_list = list(features.columns)
        features = np.array(features)


        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        results = []

        # Split the data into training and testing sets
        param = {
            'eta': 0.3,
            'max_depth': 3,
            'objective': 'multi:softprob',
            'num_class': 3}

        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rf = XGBClassifier()
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            results.append(f1_score(y_test, y_pred))

        print("Mean Absolute Error: ", np.mean(results))
        mape = 100 - round(np.mean(results), 2)
        print("Accuracy: ", mape, "%.")



        # Use the forest's predict method on the test data
        predictions = rf.predict(X_test)
        #print(predictions)
        # Calculate the absolute errors
        errors = abs(predictions - y_test)




        # Get numerical feature importances
        importances = list(rf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        # Print out the feature and importances
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

        from sklearn.metrics import recall_score, precision_score
        from sklearn.metrics import precision_recall_fscore_support as score

        recall = recall_score(y_test, predictions, average='weighted')
        precision = precision_score(y_test, predictions, average='weighted')

        print('\nPrecision:', precision)

        print('Recall:', recall)

        return rf, feature_list

        # Testing predictions (to determine performance)