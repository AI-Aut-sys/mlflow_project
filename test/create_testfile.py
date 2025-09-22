#from src.utils.input import YAMLLoader
import pandas as pd


#config = YAMLLoader().load_file("params.yaml")
#filepath = config["data"]["raw_data"]

filepath = r"D:\Users\desk\Documents\Artificial Intelligence Specialist - LEA.E3\GIT_projects\ml_Project\data\raw\cleaned_weather.csv"

df = pd.read_csv(filepath,encoding="utf-8")
sample_df =  df.sample(n=10)

sample_df.to_csv(r"D:\Users\desk\Documents\Artificial Intelligence Specialist - LEA.E3\GIT_projects\ml_Project\data\processed\small_cleaned_weather.csv", index=False) 

