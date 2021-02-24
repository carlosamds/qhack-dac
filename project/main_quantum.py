import time
from numpy import ceil 
import torch

import pennylane as qml
from pennylane import numpy as np
import random
import gym
from gym.spaces import Discrete
from gym.wrappers import Monitor

from constants import *
from helpers import ProcessObsInputEnv, ReplayBuffer, linear_schedule
from q_network import cost, qnode


if __name__ == "__main__":
    start_time = time.time()

    # setup the environment
    experiment_name = f"{ENV_NAME}__{SEED}__{int(time.time())}"

    # seeding
    device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
    env = ProcessObsInputEnv(gym.make(ENV_NAME))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = TORCH_DETERMINISTIC
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    # respect the default timelimit
    assert isinstance(env.action_space, Discrete), "only discrete action space is supported"
    if CAPTURE_VIDEO:
        env = Monitor(env, f'videos/{experiment_name}')


    q_network = np.random.normal(0, np.pi, (N_LAYERS, N_WIRES))
    target_network = q_network

    rb = ReplayBuffer(BUFFER_SIZE)
    optimizer = qml.AdamOptimizer(LR, beta1=0.9, beta2=0.999)
    print(q_network)

    # start the game
    obs = env.reset()
    episode_reward = 0
    s = time.time()
    for global_step in range(TOTAL_TIMESTEPS):
        # put action logic here
        epsilon = linear_schedule(START_EPSILON, END_EPSILON, EXPLORATION_FRACTION*TOTAL_TIMESTEPS, global_step)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            logits = qnode(q_network, obs)
            action = min(1, np.argmax(logits))

        # execute the game and log data.
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward

        # training
        rb.put((obs, action, reward, next_obs, done))
        
        if global_step > LEARNING_STARTS and global_step % TRAIN_FREQ == 0:
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(BATCH_SIZE)

            logits = np.array([qnode(target_network, obs).tolist() for obs in s_next_obses])
            target_max = np.argmax(logits, axis=1).round()
            target_max[target_max > 1] = 1

            td_target = s_rewards + GAMMA * target_max * (1 - s_dones)
    
            q_network, loss = optimizer.step_and_cost(lambda params: cost(qnode, params, s_obs, td_target, s_actions), q_network)

            if global_step % 100 == 0:
                print(f"\nloss {global_step}: {loss:.6f} | eps: {epsilon:.4f} | reward: {episode_reward:.4f} | {time.time()-s:.3f}s\n{q_network}")
                s = time.time()
    
            # update the target network
            if global_step % TARGET_NET_UPDATE_FREQ == 0:
                target_network = q_network

        # crucial step easy to overlook 
        obs = next_obs

        if done:
            print('.', end='', flush=True)
            obs, episode_reward = env.reset(), 0

    print(q_network)
    print(f"Training took {time.time() - start_time:.3f}s\n")


    # Seeing trained agent test
    for i in range(10):
        total_reward = 0
        done = False
        obs = env.reset()
        while not done:
            logits = qnode(q_network, obs)
            action = min(1, np.argmax(logits))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.1)
        print("total_reward", total_reward)

    env.close()