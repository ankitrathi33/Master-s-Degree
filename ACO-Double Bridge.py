#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter
from scipy.spatial import distance

import time
import copy
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# In[ ]:





# In[2]:


def print_matrix(m):
    for i in m:
        print(i)
    print("")

def gen_rand_coordinates():
    coordinates = [(2, 4), (4, 0), (4, 6), (6, 4), (8, 4), (10, 2), (10, 7), (12, 4)]
    return coordinates

coordinates = gen_rand_coordinates()


# In[3]:


edges_temp = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)]


# In[4]:


# edges_dict = {0: [1, 2], 1:[0, 3], 2:[0, 3], 3:[1, 2, 4], 4:[3, 5, 6], 5:[4, 7], 6:[4, 7], 7:[5, 6]}
edges_dict = {0: [1, 2], 1:[3], 2:[3], 3:[4], 4:[5, 6], 5:[7], 6:[7]}


# In[ ]:





# In[5]:


beta = 5
alpha = 1
no_of_ants = 30
distance_matrix = []
no_of_cities = len(coordinates)
pheromone_evaporation_level = 0.5

for i in range(no_of_cities):
    a = []
    for k in range(no_of_cities):
        a.append(int(distance.euclidean(coordinates[i], coordinates[k])))
    distance_matrix.append(a)

print("Distance Matrix")
print_matrix(distance_matrix)


# In[6]:


G = nx.Graph()
for p in range(len(coordinates)):
    G.add_node(p, pos=[coordinates[p][0], coordinates[p][1]])
pos = nx.get_node_attributes(G, 'pos')


# In[ ]:





# In[7]:


no_of_ants = 10
plm = [-1, -1]
history_ant_his = []
dump = []
iteration = 0

pheromone_level = list(np.zeros((no_of_cities, no_of_cities)))
while len(plm) > 1:
    ant_history = []
    for i in range(no_of_ants):
        ant_choice = []

        antchoice = 0
        distance_ = 0

        ant_choice.append(antchoice)

        while antchoice!=7:

            probs = []
            denominator = 0
            i_node = ant_choice[-1]

            allowed_cites = edges_dict[antchoice]

            for j_node in allowed_cites:
                probs.append(((pheromone_level[i_node][j_node]) ** alpha) * ((1.0/ (distance_matrix[i_node][j_node]) ** beta)))
                denominator += ((pheromone_level[i_node][j_node]) ** alpha) * ((1.0 / (distance_matrix[i_node][j_node]) ** beta))
            probs /= denominator

            if iteration == 0:
                antchoice = random.choice(allowed_cites)
            else:
                antchoice = np.random.choice(allowed_cites, p=probs)
            ant_choice.append(antchoice)
            distance_ += distance_matrix[ant_choice[len(ant_choice) - 1]][ant_choice[len(ant_choice) - 2]]

        ant_history.append((ant_choice, distance_))

    cal_edges = []
    G_dash = copy.deepcopy(G)
    for an in ant_history:
        path = an[0]
        for nod in range(len(path) - 1):
            if (path[nod], path[nod + 1]) and (path[nod + 1], path[nod]) not in cal_edges:
                cal_edges.append((path[nod], path[nod + 1]))
            elif (path[nod], path[nod + 1]) in cal_edges:
                cal_edges.append((path[nod], path[nod + 1]))
            else:
                cal_edges.append((path[nod + 1], path[nod]))

    weighted_edges = Counter(cal_edges)

    for pher_i in range(len(pheromone_level)):
        for pher_j in range(len(pheromone_level)):
            pheromone_level[pher_i][pher_j] = (1 - pheromone_evaporation_level) * pheromone_level[pher_i][pher_j]
            pheromone_level[pher_j][pher_i] = pheromone_level[pher_i][pher_j]
    
    for k in ant_history:
        l = k[1]
        path = k[0]

        for i in range(len(path) - 1):
            G_dash.add_edge(path[i], path[i + 1], weight=(weighted_edges[(path[i], path[i + 1])] + weighted_edges[(path[i + 1], path[i])]) / (no_of_ants / 4))
            pheromone_level[path[i]][path[i + 1]] += (1.0 / l)
            pheromone_level[path[i + 1]][path[i]] = pheromone_level[path[i]][path[i + 1]]

    history_ant_his = ant_history


    coun = []

    for ck in ant_history:
        coun.append(ck[1])

    plm = Counter(coun)

    average_distance = sum([i[1] for i in ant_history])/len(ant_history)
    print("Average Distance: ", average_distance)
    print("Unique Paths:", len(plm))

    dump.append([average_distance, len(plm)])

    edges = G_dash.edges()
    weights = [G_dash[u][v]['weight'] for u, v in edges]

    nx.draw_networkx(G_dash, pos, node_size=300, edges=edges, width=weights)
    plt.show()

    iteration += 1
plt.plot([i[0] for i in dump], label='Average Length')
plt.plot([i[1] for i in dump], label='Unique Paths')
plt.legend(loc="best")
plt.xlabel("Iterations")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:









# In[ ]:





# In[ ]:





# In[ ]:




