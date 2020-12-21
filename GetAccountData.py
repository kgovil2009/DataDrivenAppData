
from DataAccessObject import GetData
import json, requests
import argparse
import pandas as pd
import sys


class AccountData:
    def getAccountData():
        gd = GetData()
        features, features_high, features_low = gd.getAppData()

        # print(features)

        return features, features_high, features_low