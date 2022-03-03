import pandas as pd


old = pd.read_csv("data/old.csv")
old = old.drop(['real_lev', 'index', 'startTime'], axis=1)
new = pd.read_csv("data/new.csv")
new = new.drop(['startTime'], axis=1) 

print(old)
print(new)
print((old != new).sum())