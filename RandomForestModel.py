import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from GetAccountData import AccountData
import matplotlib.pyplot as plt

"""
Random Forest training model is implemented here. The data is gathered from GetAccountData.
"""

class RandomForestEvaluation:
    def randomForestModel(self):
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
        train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                  random_state=42)
        """
        print('Training Features Shape:', train.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test.shape)
        print('Testing Labels Shape:', test_labels.shape)
        print(test_labels)
        """
        train = train.astype('float')
        train_labels = train_labels.astype('float')
        test = test.astype('float')
        test_labels = test_labels.astype('float')
        """
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
        rf = RandomForestClassifier(n_estimators=1000, random_state=42)
        # Train the model on training data
        rf.fit(train, train_labels)

        # Use the forest's predict method on the test data
        predictions = rf.predict(test)
        # print(predictions)
        # Calculate the absolute errors
        errors = abs(predictions - test_labels)
        # print(errors)

        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2))

        # Calculate mean absolute percentage error (MAPE)
        mape = 100 - round(np.mean(errors), 2)
        #print(mape)

        # Calculate and display accuracy
        accuracy = np.mean(mape)
        print('Accuracy:', round(accuracy, 2), '%.')

        n_nodes = []
        max_depths = []

        # Stats about the trees in random forest
        for ind_tree in rf.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        #print(f'Average number of nodes {int(np.mean(n_nodes))}')
        #print(f'Average maximum depth {int(np.mean(max_depths))}')

        #print(f'Average number of nodes {int(np.mean(n_nodes))}')
        #print(f'Average maximum depth {int(np.mean(max_depths))}')

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

        recall = recall_score(test_labels, predictions, average='weighted')
        precision = precision_score(test_labels, predictions, average='weighted')

        print('\nPrecision:', precision)

        print('Recall:', recall)

        return rf, feature_list

        # Testing predictions (to determine performance)
