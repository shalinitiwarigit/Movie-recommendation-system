import numpy as np
import pandas as pd

df=pd.read_csv('ratings.csv')
df2=pd.read_csv('movies.csv')
merged_df = pd.merge(df, df2, on='movieId', how='inner')