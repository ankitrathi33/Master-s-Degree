import matplotlib.pyplot as plt

def iteration(A, population):
    new_val = [0, 0, 0]
    for i in range(0, 3): # vertical
        for j in range(0, 3): # horizontal
            new_val[j] = new_val[j] + A[i][j] * population[i]
    return new_val

# a
A = [
    [0.85, 0.07, 0.08],# plus
    [0.12, 0.73, 0.15],# play
    [0.03, 0.04, 0.93]# new mobile
]

plus = 7500000
play = 6350000
new_mobile = 7500000

population = [plus, play, new_mobile]

plus_vals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
play_vals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mobile_vals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print("")
print("ex a")
print("")
for n in range(0, 12):
    population = iteration(A, population)
    print(n)
    print(population)
    plus_vals[n] = population[0]
    play_vals[n] = population[1]
    mobile_vals[n] = population[2]

# b
print("")
print("ex b")
print("")

plus_vals2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
play_vals2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mobile_vals2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

population = [plus, play, new_mobile]

for n in range(0, 4):
    population = iteration(A, population)
    print(n)
    print(population)
    plus_vals2[n] = population[0]
    play_vals2[n] = population[1]
    mobile_vals2[n] = population[2]
print("new phone prices")
A = [
    [0.88, 0.06, 0.07],# plus +3
    [0.2, 0.69, 0.11],# play
    [0.05, 0.03, 0.92]# new mobile
]
for n in range(4, 7):
    population = iteration(A, population)
    print(n)
    print(population)
    plus_vals2[n] = population[0]
    play_vals2[n] = population[1]
    mobile_vals2[n] = population[2]

print("inflation")
A = [
    [0.905, 0.06, 0.045],# plus +3
    [0.205, 0.7, 0.095],# play
    [0.055, 0.035, 0.91]# new mobile
]
for n in range(7, 12):
    population = iteration(A, population)
    print(n)
    print(population)
    plus_vals2[n] = population[0]
    play_vals2[n] = population[1]
    mobile_vals2[n] = population[2]

fig, axs = plt.subplots(2)
fig.suptitle('ex a and ex b')

axs[0].plot(plus_vals, 'o')
axs[0].plot(play_vals, '+')
axs[0].plot(mobile_vals, '*')

axs[1].plot(plus_vals2, 'o')
axs[1].plot(play_vals2, '+')
axs[1].plot(mobile_vals2, '*')

plt.show()

