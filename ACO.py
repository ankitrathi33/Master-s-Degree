#!/usr/bin/env python
# coding: utf-8

# In[3]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from scipy.spatial import distance
from collections import Counter
import time

start_time = time.time()

def print_matrix(m):
    for i in m:
        print(i)
    print("")

def gen_rand_coordinates():
    coordinates = [(0, 3), (3, 4), (6, 5), (7, 3), (15, 0), (12, 4), (14, 10), (9, 6), (7, 9), (0, 10)]
    return coordinates

coordinates = gen_rand_coordinates()

"""
Hyperparameters
alpha: importance to pheromone
beta: importance to short distance
"""
beta = 5
alpha = 1
no_of_ants = 30
distance_matrix = []
no_of_cities = len(coordinates)
pheromone_evaporation_level = 0.5

"""
taking the l2(euclidean) distance
"""
for i in range(no_of_cities):
    a = []
    for k in range(no_of_cities):
        a.append(int(distance.euclidean(coordinates[i], coordinates[k])))
    distance_matrix.append(a)

print("Distance Matrix")
print_matrix(distance_matrix)

"""
This graph plugin shows the actual position by taking the coordinates (x, y)
"""
G = nx.Graph()
for p in range(len(coordinates)):
    G.add_node(p, pos=[coordinates[p][0], coordinates[p][1]])
pos = nx.get_node_attributes(G, 'pos')

"""
pheromone_level: we initialise the pheromone level to 0 on all the edges
plm: Number of Unique Paths by all ants

"""
pheromone_level = list(np.zeros((no_of_cities, no_of_cities)))

plm = [-1, -1]
history_ant_his = []

iteration = 0

dump = []

while len(plm) > 1:
    ant_history = []
    for i in range(no_of_ants):

        """
        We keep a track of where the ants can go. 
        Initially they can go anywhere
        But once they have gone to a edge, we remove that edge from this list (for that particular ant)
        """
        allowed_cites = list(range(no_of_cities))

        ant_choice = []
        distance = 0

        antchoice = 0
        ant_choice.append(antchoice)
        allowed_cites.remove(antchoice)

        while allowed_cites:

            denominator = 0

            i_node = ant_choice[len(ant_choice) - 1]

            probs = []

            """
            Probability is calculated by taking the the pheromone updation formula.
            Denominator is computed to normalise the values.
            """
            for j_node in allowed_cites:
                probs.append(((pheromone_level[i_node][j_node]) ** alpha) * ((1.0/ (distance_matrix[i_node][j_node]) ** beta)))
                denominator += ((pheromone_level[i_node][j_node]) ** alpha) * ((1.0 / (distance_matrix[i_node][j_node]) ** beta))

            """Normalisation"""
            probs = probs / denominator

            if iteration == 0:
                antchoice = random.choice(allowed_cites)
            else:
                antchoice = np.random.choice(allowed_cites, p=probs)
            ant_choice.append(antchoice)
            allowed_cites.remove(antchoice)
            distance += distance_matrix[ant_choice[len(ant_choice) - 1]][ant_choice[len(ant_choice) - 2]]

        ant_choice.append(ant_choice[0])
        distance += distance_matrix[ant_choice[len(ant_choice) - 1]][ant_choice[len(ant_choice) - 2]]

        ant_history.append((ant_choice, distance))
    
    G_dash = copy.deepcopy(G)

    cal_edges = []

    """
    making a note of where the ants chose to go next
    """
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

    """
    Pheromone Updation
    """
    for pher_i in range(len(pheromone_level)):
        for pher_j in range(len(pheromone_level)):
            pheromone_level[pher_i][pher_j] = (1 - pheromone_evaporation_level) * pheromone_level[pher_i][pher_j]
            pheromone_level[pher_j][pher_i] = pheromone_level[pher_i][pher_j]

    ### Drawing Edges in graph
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

print(history_ant_his[0])
print("Iterations:", iteration)
print("Time %s" % (time.time() - start_time))


# In[4]:


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





# In[ ]:




