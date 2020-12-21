import matplotlib.pyplot as plt
from RandomForestModel import RandomForestEvaluation
import numpy as np
from XGBoostModel import XGBoostEvaluation
from RandomForestKFold import RandomForestEvaluationKFold
from XGBoostKFold import XGBoostEvaluationKFold
from StatsAnalysis import StatsAnalysis

# timer calls this function after 20 seconds and closes the window
# implemented to automatically close the graph figures
def close_event():
    plt.close()


'''
class graph contains function graphComparison() : 
this has been used to display feature importance and bar graph comparison of the mean of the features of the ponzi vs non ponzi data
'''


class Graph:
    def graphComparison():
        # Get numerical feature importances

        print("\n\nStats Analysis:")
        Stat = StatsAnalysis()
        Stat.StatsMannWhiteney()

        print("\n\nRandom Forest (Test Train Split):")
        rfp = RandomForestEvaluation()
        rf, feature_list = rfp.randomForestModel()

        print("\n\nRandom Forest (KFold):")
        rfp2 = RandomForestEvaluationKFold()
        rf, feature_list = rfp2.randomForestModelkFold()

        print("\n\nXGBoost (Test Train Split):")

        xgb = XGBoostEvaluation()
        xg_feature_list = xgb.xgBoostModel()

        print("\n\nXGBoost (KFold):")

        xgbk = XGBoostEvaluationKFold()
        xg_feature_list = xgbk.xgBoostModelKFold()


    """
        print("\n\nFeature Importance\n")
        importances = list(rf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        # Print out the feature and importances
        for pair in feature_importances:
            print('Variable: {:20} Importance: {}'.format(*pair))

        print("Mean for Non Ponzi Schemes:")
        print(mean_nonponzi_data.mean(axis=0))
        print("Mean for Ponzi Schemes:")
        print(mean_ponzi_data.mean(axis=0))

        # graph comparing the features for ponzi vs non ponzi schemes
        x = mean_ponzi_data.mean(axis=0, skipna=True)
        y = mean_nonponzi_data.mean(axis=0, skipna=True)

        x = np.array(x)
        y = np.array(y)

        indices = [5.0, 7, 9, 11, 13]  # the x locations for the groups
        width = np.min(np.diff(indices)) / 3

        fig = plt.figure()
        timer = fig.canvas.new_timer(
            interval=90000)  # creating a timer object and setting an interval of 20000 milliseconds
        timer.add_callback(close_event)
        ax = fig.add_subplot(111)
        ax.bar(indices, x, width)
        ax.bar(indices + width, y, width)

        ax.set_ylabel('Values')
        ax.set_title('Mean')
        ax.set_xticks(indices + width / 2)
        ax.set_xticklabels(('Bal', 'N_maxpay', 'N_Investment', 'N_Payment', 'Paid_rate'))

        timer.start()
        plt.show()"""
