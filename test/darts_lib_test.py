from darts.datasets import AirPassengersDataset
import pandas as pd

series = AirPassengersDataset().load()
print(series.head())

print(f"Length of dataset: {len(series)}")
print(f"Time series start: {series.start_time()}")
print(f"Time series end: {series.end_time()}")
df = series.to_dataframe()
print(df.head())
print(df.info())
print(df.shape)