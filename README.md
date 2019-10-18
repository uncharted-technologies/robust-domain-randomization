# Domain Randomization
Code associated with our paper "Robust Domain Randomization for Reinforcement Learning"

# Requirements

Requirements can be found in the requirements.txt file.

# Structure
The project contains one folder per domain studied in the main text of our paper: the gridworld, the visual cartpole domain, and Car Racing. Each folder contains the following subfolders:

**agent: the learning agent: DQN, Vanilla Policy Gradient (VPG), or PPO

**experiments: containing scripts to reproduce the experiments reported in our paper.

**results: by default, the output of scripts from the experiments folder is saved here

**plotting: loads data from the results folder and plots it, yielding the figures in our paper

# Running experiments: example
To train a PPO agent on the CarRacing environment, run the following command from the car_racing folder:

>>> python -m experiments.train_ppo

A log folder will automatically be created in the results folder, which at the end of training will contain the network weights of the trained agent. The agent's generalization ability can then be tested with the following command:

>>> python -m experiments.test_generalization

The agent's training curve can also be generated with

>>> python -m plotting.plot_train

Note that we have not included the weights of the trained agents that we used in our paper for the car racing and visual cartpole experiments in this repository since this would cause this repository to be unreasonable large. Some scripts in this repository (such as those used for testing generalization) do require trained agents; you can either train them yourself using the trainings scripts provided or contact us and we can provide them upon reasonable request.

# Convention
In this project, we compare three agents : *Normal*, *Regularized* and *Randomized* (see article for details). Each agent corresponds to a combinaison of booleans to modify in the start of the scripts for the visual cartpole and gridworld experiments:
- *Normal* : RANDOMIZE = False, REGULARIZE = False
- *Regularized* : RANDOMIZE = True, REGULARIZE = True
- *Randomized* : RANDOMIZE = True, REGULARIZE = False

