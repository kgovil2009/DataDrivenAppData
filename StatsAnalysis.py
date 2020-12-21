from GetAccountData import AccountData
from scipy import stats


class StatsAnalysis:
    def StatsMannWhiteney(self):
        features, features_high_rating, features_low_rating = AccountData.getAccountData()

        u_statistic_1, p_value_1 = stats.mannwhitneyu(list(features_high_rating['Promotional_images']), list(features_low_rating['Promotional_images']))
        print("Promotional Images p_value:",p_value_1)


        u_statistic_2, p_value_2 = stats.mannwhitneyu(list(features_high_rating['Category']), list(features_low_rating['Category']))
        print("Category p_value:",p_value_2)


        u_statistic_3, p_value_3 = stats.mannwhitneyu(list(features_high_rating['description_id']), list(features_low_rating['description_id']))
        print("Description p_value:",p_value_3)


        u_statistic_4, p_value_4 = stats.mannwhitneyu(list(features_high_rating['Size_Category']), list(features_low_rating['Size_Category']))
        print("Size Category p_value:",p_value_4)


        u_statistic_5, p_value_5 = stats.mannwhitneyu(list(features_high_rating['Installs']), list(features_low_rating['Installs']))
        print("Installs p_value:",p_value_5)


        u_statistic_6, p_value_6 = stats.mannwhitneyu(list(features_high_rating['Content_rating']), list(features_low_rating['Content_rating']))
        print("Content Rating p_value:",p_value_6)


        u_statistic_7, p_value_7 = stats.mannwhitneyu(list(features_high_rating['Type']), list(features_low_rating['Type']))
        print("Type p_value:",p_value_7)


        u_statistic_8, p_value_8 = stats.mannwhitneyu(list(features_high_rating['Min_SDK_Version']), list(features_low_rating['Min_SDK_Version']))
        print("SDK Version p_value:",p_value_8)

