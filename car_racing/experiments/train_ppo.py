"""
This code is inspired by https://github.com/nikhilbarhate99/PPO-PyTorch
"""


from agent.ppo_simple import Memory, PPO
from carracing_wrapper import make_carracing
import gym
import numpy as np
from agent.logger import Logger
import torch

def main():
    ############## Hyperparameters ##############
    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = 1000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    
    update_timestep = 1000      # update policy every n timesteps
    action_std = 0.2         # constant std for action distribution (Multivariate Normal)
    K_epochs = 20               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0001                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None

    color_range = [0,0.5]
    randomize = False
    lamda = 0
    #############################################

    train_details = {
        'max_episodes':max_episodes,
        'update_timestep':update_timestep,
        'action_std':action_std,
        'K_epochs':K_epochs,
        'eps_clip':eps_clip,
        'gamma':gamma,
        'learning_rate':lr,
        'randomize':randomize,
        'lamda':lamda,
        'color_range':color_range
    }
    
    # creating environment
    
    env = make_carracing('CarRacing-v0',color_range)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip,randomize,lamda)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    logger = Logger(train_details=train_details)
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        this_episode_score = 0
        state,state_randomized = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, state_randomized, memory)

            state,state_randomized, reward, done, _ = env.step(action)
            this_episode_score += reward
            # Saving reward:
            memory.rewards.append(reward)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
            running_reward += reward
            if render:
                env.render()
            if done:
                logger.add_scalar('Episode_score',this_episode_score, time_step)
                this_episode_score = 0
                break
        
        avg_length += t
        
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), logger.log_folder + '/PPO_continuous.pth')
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

    logger.save()
            
main()