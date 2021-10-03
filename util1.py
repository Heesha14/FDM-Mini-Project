import pickle

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv("dataset_group.csv", header=None)
df.columns = ["Date", "ID", "Items"]

df.drop('Date', inplace=True, axis=1)

df1 = df.groupby('ID')['Items'].apply(','.join).reset_index()
pd.set_option('display.max_rows', df.shape[0] + 1)

transac = []
for i in range(0, len(df1)):
    transac.append([str(df1.values[i, j]) for j in range(0, 2) if str(df1.values[i, j]) != '0'])

itemArray = df['Items'].unique()

df = df.groupby('ID')['Items'].apply(','.join).reset_index()

for i in range(0, len(itemArray)):
    df.insert(len(df.columns), itemArray[i], "")
    for index, row in df.iterrows():
        df.at[index, itemArray[i]] = 1 if len(
            [1 for item in transac[index][1].split(",") if item in itemArray[i]]) > 0 else 0

df.drop('Items', inplace=True, axis=1)

frequent_itemsets = apriori(df.drop(['ID'], axis=1), min_support=0.2, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

#saving model to disk
pickle.dump(rules, open('rules.pkl', 'wb'))