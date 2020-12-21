from configuration import cfg as c
import pyodbc as p
import pandas as pd

"""This file takes the credentials from configuration.py and connects with Amazon RDS instance. The sql server db 
instance has the data for high rated and low rated apps. """


connection = p.connect('Driver={SQL Server}',
                      Server=c.server,
                      Database=c.dbname,
                      UID=c.user,
                      PWD=c.password
                      )

class GetData:
    def getAppData(self):
        df = pd.DataFrame(
            columns=['ID', 'Category', 'Rating_id', 'description_id', 'Size_Category', 'Reviews', 'Installs', 'Type', 'Content_rating', 'Promotional_images', 'Min_SDK_Version'])
        query = 'SELECT ID, Category, Rating_id, description_id, Size_Category, Reviews, Installs, Type, Content_rating, Promotional_images, Min_SDK_Version FROM appData.dbo.PlayStoreMaster'
        df = pd.read_sql(query, connection)

        df_high = pd.DataFrame(
            columns=['ID', 'Category', 'Rating_id', 'description_id', 'Size_Category', 'Reviews', 'Installs', 'Type', 'Content_rating', 'Promotional_images', 'Min_SDK_Version'])
        query = 'SELECT ID, Category, Rating_id, description_id, Size_Category, Reviews, Installs, Type, Content_rating, Promotional_images, Min_SDK_Version FROM appData.dbo.PlayStoreMaster where Rating_id = 1'
        df_high = pd.read_sql(query, connection)

        df_low = pd.DataFrame(
            columns=['ID', 'Category', 'Rating_id', 'description_id', 'Size_Category', 'Reviews', 'Installs', 'Type', 'Content_rating', 'Promotional_images', 'Min_SDK_Version'])
        query = 'SELECT ID, Category, Rating_id, description_id, Size_Category, Reviews, Installs, Type, Content_rating, Promotional_images, Min_SDK_Version FROM appData.dbo.PlayStoreMaster where Rating_id = 1'
        df_low = pd.read_sql(query, connection)

        return df, df_high, df_low



