import matplotlib.pyplot as plt
import numpy as np


res = np.load('results/std_valueFunction.npy')

plt.plot(list(range(res.shape[1])), res[0], label = 'Normal, mean = '+ str(round(np.mean(res[0]),2)), color = 'chocolate')
plt.plot(list(range(res.shape[1])), res[1], label = 'Reg, mean = '+ str(round(np.mean(res[1]),2)), color = 'blue')
plt.plot(list(range(res.shape[1])), res[2], label = 'Rand, mean = ' + str(round(np.mean(res[2]),2)), color = 'green')

plt.legend()
plt.title('Std of the value function over domains averaged over one episode')
plt.xlabel('Seed')
plt.ylabel('Std of value function')
plt.show()
