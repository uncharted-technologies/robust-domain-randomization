import pickle
import numpy as np

class TestResults():
    def __init__(self, agent, agent_type):
        self.agent = agent
        self.agent_type = agent_type
        self.returns = []
        self.colors = []
    def add_return(self,returns,color):
        self.returns.append(returns)
        self.colors.append(color)

results = pickle.load(open('results/generalization_results_random.pkl','rb'))

normal_results = []
randomized_results = []
regularized_results = []

for test_results in results:

    if test_results.agent_type == 'normal':
        normal_results.append(np.mean(test_results.returns))
    elif test_results.agent_type == 'regularize':
        regularized_results.append(np.mean(test_results.returns))
    else:
        randomized_results.append(np.mean(test_results.returns))

print('Normal agent: ', np.array(normal_results).mean(),1.96*np.array(normal_results).std()/np.sqrt(5))
print('Regularized agent: ', np.array(regularized_results).flatten().mean(),1.96*np.array(regularized_results).std()/np.sqrt(5))
print('Randomized agent: ', np.array(randomized_results).flatten().mean(),1.96*np.array(randomized_results).std()/np.sqrt(5))