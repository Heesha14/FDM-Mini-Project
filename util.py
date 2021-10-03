import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import networkx as nx

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def recommend_product(item_list):
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

    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent))
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent))
    rules['rule'] = rules.index
    print(rules['antecedent'])

    cons_list = []
    for i in range(1, len(rules)):
        check = all(item in (rules['antecedent'][i]) for item in item_list)
        if check is True:
            c_list = set(list(rules['consequents'])[i])
            cons_list.append(c_list)
    print(cons_list)
    #print(cons_list[0])
    #print(" , ".join(cons_list[4]))

    return cons_list


def parallelPlot():
    df = pd.read_csv("dataset_group.csv", header=None)
    df.columns = ["Date", "ID", "Items"]

    df.drop('Date', inplace=True, axis=1)

    df1 = df.groupby('ID')['Items'].apply(','.join).reset_index()

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

    frequent_itemsets = apriori(df.drop(['ID'], axis=1), min_support=0.3, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
    rules['rule'] = rules.index
    return rules[['antecedent', 'consequent', 'rule']]


def rule_list():
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

    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent))
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent))
    rules['rule'] = rules.index

    return rules



def networkXplot(rules, rules_to_show):
    G1 = nx.DiGraph()

    color_map=[]
    N = len(rules)
    colors = np.random.rand(N)
    strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11','R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23','R24', 'R25', 'R26', 'R27', 'R28', 'R29', 'R30', 'R31', 'R32', 'R33', 'R34', 'R35','R36', 'R37']
    for i in range (rules_to_show):
        G1.add_nodes_from(["R"+str(i)])

        for a in rules.iloc[i]['antecedents']:

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:

                G1.add_nodes_from([c])

                G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)

    for node in G1:
        found_a_string = False
        for item in strs:
            if node==item:
                found_a_string = True
        if found_a_string:
            color_map.append('yellow')
        else:
            color_map.append('green')

    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]

    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos,node_color = color_map, edge_color=colors, width=weights,font_color='white',with_labels=False)

    for p in pos:  # raise text positions
        pos[p][1] += 0.07
    nx.draw_networkx_labels(G1, pos)
    #plt.figure(3,figsize=(4,8))
    plt.title('NetworkX Plot')
    plt.savefig('static/' + 'plot2.png', dpi=600, edgecolor="#04253a")
    plt.close()


def networkPlotRule(rules, rules_to_show,item_list):
    G1 = nx.DiGraph()

    color_map=[]
    N = len(rules)
    colors = np.random.rand(N)
    strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11','R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23','R24', 'R25', 'R26', 'R27', 'R28', 'R29', 'R30', 'R31', 'R32', 'R33', 'R34', 'R35','R36', 'R37']
    for i in range (rules_to_show):
        j = 0
        check = all(item in (rules['antecedent'][i]) for item in item_list)
        if check is True:
            G1.add_nodes_from(["R"+str(j)])
        else:
            continue

        for a in rules.iloc[i]['antecedent']:
            check = all(item in (rules['antecedent'][i]) for item in item_list)
            if check is True:
                G1.add_nodes_from([a])
                G1.add_edge(a, "R"+str(j), color=colors[j] , weight = 2)
            else:
                continue

        for c in rules.iloc[i]['consequents']:
            check2 = all(item in (rules['antecedent'][i]) for item in item_list)
            if check2 is True:
                G1.add_nodes_from([c])
                G1.add_edge("R"+str(j), c, color=colors[j],  weight=2)
            else:
                continue
        j += 1

    for node in G1:
        found_a_string = False
        for item in strs:
            if node==item:
                found_a_string = True
        if found_a_string:
            color_map.append('yellow')
        else:
            color_map.append('green')

    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]


    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos,node_color = color_map, edge_color=colors, width=weights,font_color='white',with_labels=False)
    for p in pos:  # raise text positions
        pos[p][1] += 0.07
    nx.draw_networkx_labels(G1, pos)
    plt.title('NetworkX Plot')
    plt.savefig('static/' + 'plot3.png', dpi=600, edgecolor="#04253a")
    plt.close()


