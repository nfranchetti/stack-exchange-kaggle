import pandas as pd
import numpy as np
import os


bashCommand = "sed 1d submissions/submission*.csv > submissions/merged.csv"
os.system(bashCommand)

df = pd.read_csv('submissions/merged.csv', names=['id', 'tags'])

print df.head()
print df.columns

df.drop(df[(df['id'] == 'id')], inplace=True)
print df.shape