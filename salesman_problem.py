

import math
import numpy as np
import random
import matplotlib.pyplot as plt


n=0.8;
class City:
   def __init__(self, x=None, y=None):
      self.x = None
      self.y = None
      if x is not None:
         self.x = x
      else:
         self.x = int(random.random() * 200)
      if y is not None:
         self.y = y
      else:
         self.y = int(random.random() * 200)
   
   def getX(self):
      return self.x
   
   def getY(self):
      return self.y
   
   def distanceTo(self, city):
      xDistance = abs(self.getX() - city.getX())
      yDistance = abs(self.getY() - city.getY())
      distance = math.sqrt( (xDistance*xDistance) + (yDistance*yDistance) )
      return distance
   
   def __repr__(self):
      return str(self.getX()) + ", " + str(self.getY())


class TourManager:
   destinationCities = []
   
   def addCity(self, city):
      self.destinationCities.append(city)
   
   def getCity(self, index):
      return self.destinationCities[index]
   
   def numberOfCities(self):
      return len(self.destinationCities)


class Tour:
   def __init__(self, tourmanager, tour=None):
      self.tourmanager = tourmanager
      self.tour = []
      self.fitness = 0.0
      self.distance = 0
      if tour is not None:
         self.tour = tour
      else:
         for i in range(0, self.tourmanager.numberOfCities()):
            self.tour.append(None)
   
   def __len__(self):
      return len(self.tour)
   
   def __getitem__(self, index):
      return self.tour[index]
   
   def __setitem__(self, key, value):
      self.tour[key] = value
   
   def __repr__(self):
      geneString = "|"
      for i in range(0, self.tourSize()):
         geneString += str(self.getCity(i)) + "|"
      return geneString
   
   def generateIndividual(self):
      for cityIndex in range(0, self.tourmanager.numberOfCities()):
         self.setCity(cityIndex, self.tourmanager.getCity(cityIndex))
      random.shuffle(self.tour)
   
   def getCity(self, tourPosition):
      return self.tour[tourPosition]

   def getTour(self):
      return self.tour
   
   def setCity(self, tourPosition, city):
      self.tour[tourPosition] = city
      self.fitness = 0.0
      self.distance = 0
   
   def getFitness(self):
      if self.fitness == 0:
         self.fitness = 1/float(self.getDistance())
      return self.fitness
   
   def getDistance(self):
      if self.distance == 0:
         tourDistance = 0
         for cityIndex in range(0, self.tourSize()):
            fromCity = self.getCity(cityIndex)
            destinationCity = None
            if cityIndex+1 < self.tourSize():
               destinationCity = self.getCity(cityIndex+1)
            else:
               destinationCity = self.getCity(0)
            tourDistance += fromCity.distanceTo(destinationCity)
         self.distance = tourDistance
      return self.distance
   
   def tourSize(self):
      return len(self.tour)
   
   def containsCity(self, city):
      return city in self.tour


class Population:
   def __init__(self, tourmanager, populationSize, initialise):
      self.tours = []
      for i in range(0, populationSize):
         self.tours.append(None)
      
      if initialise:
         for i in range(0, populationSize):
            newTour = Tour(tourmanager)
            newTour.generateIndividual()
            self.saveTour(i, newTour)
      
   def __setitem__(self, key, value):
      self.tours[key] = value
   
   def __getitem__(self, index):
      return self.tours[index]
   
   def saveTour(self, index, tour):
      self.tours[index] = tour
   
   def getTour(self, index):
      # print(Tour(self.tours[index]).getTour())
      return self.tours[index]
   
   def getFittest(self):
      fittest = self.tours[0]
      for i in range(0, self.populationSize()):
         if fittest.getFitness() <= self.getTour(i).getFitness():
            fittest = self.getTour(i)

      return fittest
   
   def populationSize(self):
      return len(self.tours)


class GA:
   def __init__(self, tourmanager):
      self.tourmanager = tourmanager
      self.mutationRate = 0.2
      self.tournamentSize = 5
      self.elitism = True
   
   def evolvePopulation(self, pop):
      newPopulation = Population(self.tourmanager, pop.populationSize(), False)
      elitismOffset = 0
      if self.elitism:
         newPopulation.saveTour(0, pop.getFittest())
         elitismOffset = 1
      
      for i in range(elitismOffset, math.floor(newPopulation.populationSize()*0.8)):
         parent1 = self.tournamentSelection(pop)
         parent2 = self.tournamentSelection(pop)
         child = self.crossover(parent1, parent2)
         newPopulation.saveTour(i, child)

      for i in range(math.floor(newPopulation.populationSize()*0.8), newPopulation.populationSize()):
          newPopulation.saveTour(i, pop[i])
      
      for i in range(elitismOffset, newPopulation.populationSize()):
         self.mutate(newPopulation.getTour(i))
      
      return newPopulation
   
   def crossover(self, parent1, parent2):
      child = Tour(self.tourmanager)
      
      startPos = int(random.random() * parent1.tourSize())
      endPos = int(random.random() * parent1.tourSize())
      
      for i in range(0, child.tourSize()):
         if startPos < endPos and i > startPos and i < endPos:
            child.setCity(i, parent1.getCity(i))
         elif startPos > endPos:
            if not (i < startPos and i > endPos):
               child.setCity(i, parent1.getCity(i))
      
      for i in range(0, parent2.tourSize()):
         if not child.containsCity(parent2.getCity(i)):
            for ii in range(0, child.tourSize()):
               if child.getCity(ii) == None:
                  child.setCity(ii, parent2.getCity(i))
                  break
      
      return child
   
   def mutate(self, tour):
      for tourPos1 in range(0, tour.tourSize()):
         if random.random() < self.mutationRate:
            tourPos2 = int(tour.tourSize() * random.random())
            
            city1 = tour.getCity(tourPos1)
            city2 = tour.getCity(tourPos2)
            
            tour.setCity(tourPos2, city1)
            tour.setCity(tourPos1, city2)
   
   def tournamentSelection(self, pop):
      tournament = Population(self.tourmanager, self.tournamentSize, False)
      for i in range(0, self.tournamentSize):
         randomId = int(random.random() * pop.populationSize())
         tournament.saveTour(i, pop.getTour(randomId))
      fittest = tournament.getFittest()
      return fittest



if __name__ == '__main__':
   
   tourmanager = TourManager()
   
   # Create and add our cities
   city = City(0, 1)
   tourmanager.addCity(city)
   city2 = City(3, 4)
   tourmanager.addCity(city2)
   city3 = City(6, 5)
   tourmanager.addCity(city3)
   city4 = City(7, 3)
   tourmanager.addCity(city4)
   city5 = City(15, 0)
   tourmanager.addCity(city5)
   city6 = City(12, 4)
   tourmanager.addCity(city6)
   city7 = City(14, 10)
   tourmanager.addCity(city7)
   city8 = City(9, 6)
   tourmanager.addCity(city8)
   city9 = City(7, 9)
   tourmanager.addCity(city9)
   city10 = City(0, 10)
   tourmanager.addCity(city10)

   
   # Initialize population
   pop = Population(tourmanager, 250, True);
   print("Initial distance: " + str(pop.getFittest().getDistance()))

   # Evolve population for 50 generations
   ga = GA(tourmanager)
   pop = ga.evolvePopulation(pop)
   for i in range(0, 1000):
       pop = ga.evolvePopulation(pop)

   # Print final results
   tour =  pop.getFittest().getTour()

   print(tour)
   print("Finished")
   print("Final distance: " + str(pop.getFittest().getDistance()))
   print("Solution:")
   print(pop.getFittest())

   # print(pop.getTour())
iterator = 1
ourList = []
list = []





graph = [(20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 20)]

# draw_graph(graph)
#
#
#



def draw_point():
   # Draw a point at the location (3, 9) with size 1000
   i =0
   for item in tour:
      if i == 9:
         plt.scatter(item.getX(), item.getY(), s=100)
         x1, x2 = item.getX(), tour[0].getX()
         y1, y2 = item.getY(), tour[0].getY()
         plt.plot([x1, x2], [y1, y2], 'k-')
         break

      plt.scatter(item.getX(), item.getY(), s=100)
      x1, x2 = item.getX(), tour[i+1].getX()
      y1, y2 = item.getY(), tour[i+1].getY()
      plt.plot([x1,x2],[y1,y2],'k-')

      i+=1

   # Set chart title.
   plt.title("Square Numbers", fontsize=17)

   # Set x axis label.
   plt.xlabel("Number", fontsize=10)

   # Set y axis label.
   plt.ylabel("Square of Number", fontsize=10)

   # Set size of tick labels.
   plt.tick_params(axis='both', which='major', labelsize=9)

   # Display the plot in the matplotlib's viewer.
   plt.show()


if __name__ == '__main__':
   draw_point()