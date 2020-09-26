import numpy as np 
import itertools 
import matplotlib.pyplot as plt 
import csv 
from collections import Counter 
 
number_of_mixtures = np.random.randint(40, 80) 
number_of_categories = np.random.randint(7, 20) 
number_of_observations = np.random.randint(80, 150) 
probabilities_of_categories = np.zeros((number_of_mixtures, number_of_categories)) 
 
def writeToCSV(name: str): 
 with open('{}.csv'.format(name), 'w', newline='') as file: 
     writer = csv.writer(file) 
     writer.writerow(["number_of_mixtures", number_of_mixtures]) 
     writer.writerow(["number_of_categories", number_of_categories]) 
     writer.writerow(["number_of_observations", number_of_observations]) 
 
def obsHistogram(): 
 for i in range(number_of_mixtures): 
   a = np.random.uniform(size=number_of_categories) 
   a /= a.sum() 
   probabilities_of_categories[i] = a 
 
 multinom = np.random.multinomial(number_of_observations, probabilities_of_categories[0], 
size=number_of_mixtures) 
 counter = Counter(i for i in list(itertools.chain.from_iterable(multinom))) 
 labels, values = zip(*sorted(counter.items())) 
 
 indexes = np.arange(len(labels)) 
 width = 0.9 
 plt.bar(indexes, values, width) 
 plt.xticks(indexes + width * 0.5, labels) 
 plt.show() 
 
def mixture(merge: bool): 
 for i in range(number_of_categories): 
   for i in range(number_of_mixtures): 
     a = np.random.uniform(size=number_of_categories) 
     a /= a.sum() 
     probabilities_of_categories[i] = a 

   multinom = np.random.multinomial(number_of_observations, probabilities_of_categories[i], 
size=number_of_mixtures) 
   with open('multinom.csv', 'w', newline='') as file: 
     writer = csv.writer(file) 
     for multi in multinom: 
       writer.writerow(multi) 
   indexes = np.arange(len(probabilities_of_categories[i])) 
   width = 0.9 
   plt.bar(indexes, multinom[i], width) 
   if not merge: plt.show() 
 if merge: plt.show() 
 
writeToCSV('exported_data') 
writeToCSV('exported_data') 
obsHistogram() 
mixture(True) 
# mixture(False) 




 


