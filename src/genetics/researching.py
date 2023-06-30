import pandas as pd

df = pd.DataFrame({"col1": [1,2], "col2": [2,4]})
df.dropna(axis=0, how=all)
df.fillna()