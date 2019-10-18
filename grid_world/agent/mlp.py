import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m, gain):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, gain)

class MLP_Multihead(torch.nn.Module):
    def __init__(self, observation_space, n_outputs_1, n_outputs_2, width=100):

        super().__init__()

        if len(observation_space.shape) != 1:
            raise NotImplementedError
        else:
            n_inputs = observation_space.shape[0]

        self.output_1 = nn.Sequential(
                            torch.nn.Linear(n_inputs, width),
                            nn.ReLU(),
                            nn.Linear(width, width),
                            nn.ReLU(),
                            nn.Linear(width, n_outputs_1))

        self.output_2 = nn.Sequential(
                            torch.nn.Linear(n_inputs, width),
                            nn.ReLU(),
                            nn.Linear(width, width),
                            nn.ReLU(),
                            nn.Linear(width, n_outputs_2))

        #self.layer1.apply(lambda x: init_weights(x, 3))
        self.output_1.apply(lambda x: init_weights(x, 0.5))
        self.output_2.apply(lambda x: init_weights(x, 0.5))

    def forward(self, obs):
        #out = F.relu(self.layer1(obs))
        return self.output_1(obs), self.output_2(obs)

