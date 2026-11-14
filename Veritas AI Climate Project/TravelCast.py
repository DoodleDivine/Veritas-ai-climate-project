import pandas as pd

df = pd.read_csv("MonthlyWeatherDataTokyo_2010_2020.csv", skiprows=11)

print(df)