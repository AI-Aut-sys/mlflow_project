import pandas as pd
from src.utils.input import YAMLLoader


class DataTransform:
    """
    This is a class where we do feature selection, datat transfromation and aggregation
    on a dataset based on the values set in the YAML file

    """

    config = YAMLLoader().load_file("params.yaml")
    
    def __init__(self,df:pd.DataFrame,config:dict):
        """
        Initialize the DataTransform

        Args:
            df (pd.DataFrame): raw dataframe
            config (dict): This is the YAML file that contain all the setting
        """
        self.config = config
        self.df = df.copy()

    def select_features(self) -> pd.DataFrame :
        """
        select specific columns from the dataset 

        Returns:
            pd.DataFrame: it will return a dataframe with all the selected feture
        """
        final_feature = self.config["feature"]["select_feature"]
        self.df = self.df[final_feature]
        return self.df

    def transform_date(self) -> pd.DataFrame:
        """
        Since the "date" column is in object format, we have to convert into datetime64[ns],
        in order to do a time series prediction.  Other columns we added as it will be useful
        when you do grouping

        Adds the following columns:
        - 'date_dt': datetime object
        - 'minute': minute of the hour
        - 'hour': hour of the day
        - 'day': day of the month
        - 'month': month of the year
        - 'minutes_interval': "A" for minutes 0 to 30, "B" for minutes > 30

        Returns:
            pd.DataFrame: This dataframe will contain all tranform featured and new column
        """
        self.df["date_dt"] = pd.to_datetime(self.df["date"],format="%Y-%m-%d %H:%M:%S")
        self.df["minute"] = self.df["date_dt"].dt.minute
        self.df["hour"] = self.df["date_dt"].dt.hour
        self.df["day"] = self.df["date_dt"].dt.day
        self.df["month"] = self.df["date_dt"].dt.month
        self.df["minutes_interval"] = self.df["minute"].apply(lambda x:"A" if x<=30 else "B")
        return self.df
    
    def group_by(self) -> pd.DataFrame:
        """
        group_by _summary: we grouped the dataset by month, day, hour and interval of 30 mins(A/B)
        that aggregate the specified feature based on the config

        While doing aggregate, we didnot use the "date" feature, so we create a date_range to 
        serve as a datetime index and we concatenate the 2 dataframe after.

        Returns:
            pd.DataFrame: A time series dataframe 
        """
        agg_config = self.config["transform"]["aggregate"]
        df_group = self.df.groupby(["month","day","hour","minutes_interval"]).agg(agg_config).reset_index()
        date_range = pd.date_range(start="2020-01-01 00:00:00",end="2020-12-31 22:00:00",freq="30 min")
        df_ts = pd.DataFrame(date_range)
        df_ts.columns = ["date"]
        self.df = pd.concat([df_ts,df_group],axis=1)
        return self.df

