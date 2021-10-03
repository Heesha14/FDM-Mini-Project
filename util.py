import pickle

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def recommend_product(item_list):
    rules = pickle.load(open('rules.pkl', 'rb'))
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
    return cons_list

def parallelPlot():
    rules = pickle.load(open('rules.pkl', 'rb'))

    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
    rules['rule'] = rules.index
    return rules[['antecedent', 'consequent', 'rule']]

def rule_list():
    rules = pickle.load(open('rules.pkl', 'rb'))
    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent))
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent))
    rules['rule'] = rules.index

    return rules

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


