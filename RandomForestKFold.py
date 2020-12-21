import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import roc_auc_score, plot_roc_curve, f1_score

from GetAccountData import AccountData
import matplotlib.pyplot as plt

"""
Random Forest training model is implemented here. The data is gathered from GetAccountData.
"""

class RandomForestEvaluationKFold:
    def randomForestModelkFold(self):
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
        """features = features.drop('ID', axis=1)
        features = features.drop('Reviews', axis=1)"""
        # Saving feature names for later use
        feature_list = list(features.columns)
        # Convert to numpy array


        # Split the data into training and testing sets
        """
        print('Training Features Shape:', train.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test.shape)
        print('Testing Labels Shape:', test_labels.shape)
        print(test_labels)
        
        train = train.astype('float')
        train_labels = train_labels.astype('float')
        test = test.astype('float')
        test_labels = test_labels.astype('float')
        print('Train:')
        print(train)
        print('Train Labels:')
        print(train_labels)
        
        print('Test:')
        print(test)
        print('Test Labels:')
        print(test_labels)
        """
    # Instantiate model with 1000 decision trees
        # Train the model on training data

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        results = []

        for train_index, test_index in kf.split(features):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rf = RandomForestClassifier(n_estimators=100)
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
        # print(errors)

        # Print out the mean absolute error (mae)
        # print('Mean Absolute Error:', round(np.mean(errors), 2))

        # Calculate mean absolute percentage error (MAPE)
        # mape = 100 - round(np.mean(errors), 2)
        # print(mape)

        # Calculate and display accuracy
        # accuracy = np.mean(mape)
        # print('Accuracy:', round(accuracy, 2), '%.')

        n_nodes = []
        max_depths = []

        # Stats about the trees in random forest
        for ind_tree in rf.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)
        """
        print(f'Average number of nodes {int(np.mean(n_nodes))}')
        print(f'Average maximum depth {int(np.mean(max_depths))}')

        print(f'Average number of nodes {int(np.mean(n_nodes))}')
        print(f'Average maximum depth {int(np.mean(max_depths))}')
        """
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
