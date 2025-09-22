import pandas as pd
from typing import Tuple, List, Dict
from src.utils.input import YAMLLoader
from darts.models import ARIMA,NaiveMean,NaiveDrift
from darts import TimeSeries
from darts.utils.model_selection import train_test_split
from darts.metrics import smape,mse



class DataModelling():
    """
    This class will preprocess, split and model the timeseries data using darts.

    """

    config = YAMLLoader().load_file("params.yaml")

    def __init__(self,df:pd.DataFrame,config:dict):
        """
        Initialize the Datamodelling class with a dataframe and config

        Args:
            df (pd.DataFrame): The dataset 
            config (dict): Configuration from the Yaml file
        """
        self.config = config
        self.df = df.copy()

    def df_timeseries(self) -> TimeSeries:
        """
        convert the date columns into a timeseries column

        Returns:
            TimeSeries: A Timeseries dataframe
        """
        time_cols = self.config["feature"]["time_col"]
        value_cols = self.config["feature"]["value_cols"]
        return TimeSeries.from_dataframe(df=self.df,time_col=time_cols,value_cols=value_cols)
    
    def split_timeseries(self,series:TimeSeries)-> Tuple:
        """
        This function split the datatset into 2 sets , train  and test dataset

        Args:
            series (TimeSeries): The whole Timeseries dataset

        Returns:
            tuple: A tuple (X_train, X_test)
        """
        test_size = self.config["prep"]["test_size"]
        X_train,X_test = train_test_split(data=series,test_size=test_size)
        return X_train,X_test
    

    def train_model(self,X_train,X_test) -> list[Dict[str, float]]:
        """
        This function will run a list of model being train from the dataset and 
        evaluate them on the test dataset

        Args:
            X_train (TimeSeries): Train dataset
            X_test (TimeSeries): Test dataset

        Returns:
            _type_: Return a list with all the name of the models and their metrics
        """

        results = []

        models = {
                "NaiveMean": NaiveMean(),
                "NaiveDrift": NaiveDrift(),
                "ARIMA": ARIMA()
                }

        for name, model in models.items():
            model.fit(X_train)
            pred = model.predict(len(X_test))

            results.append({"Model":name,"SMAPE":smape(X_test,pred),"MSE":mse(X_test,pred)})

        return results

