import sys
sys.path.append('../')
from Network import Network
from Environment import CyberAttack
import pandas as pd
import numpy as np
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import stable_baselines3
from stable_baselines3 import PPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import joblib

network = Network(1000,random_state=0).network

def mask_fn(env):
    """
    Helper function to mask reinforcement learning actions
    """
    a_out = np.array(env.state["accessible_out"],dtype=bool)
    a_in = np.array(env.state["accessible_in"],dtype=bool)
    c = np.array(env.state["compromised"],dtype=bool)
    mask = ~(~(a_out|a_in)|c)
    mask = np.array(mask,dtype=np.int32)
    return mask

def sample_ppo_params(trial):
    """
    Input:
        trial : optuna.Trial
    Output:
        params : dict
        
    params dictionary contains the following PPO hyperparameters
        learning_rate : float - learning rate ranges from 1e-5 to 1
        n_steps : int - number of steps to run per update
        batch_size : int - minibatch size
        gamma : float - discount factor
        clip_range : float - clipping parameter
    """
    # Sample PPO hyperparameters
    learning_rate = trial.suggest_float("learning_rate",1e-5,1.,log=True)
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    if batch_size > n_steps:
        batch_size = n_steps
    # Format as dictionary
    params = {"learning_rate": learning_rate,
              "n_steps": n_steps,
              "batch_size": batch_size,
              "gamma": gamma,
              "clip_range": clip_range}
    # Return hyperparameters
    return params

def objective(trial):
    """
    Input:
        trial : optuna.Trial
    Output:
        final_reward : float
    """
    # Sample PPO hyperparameters
    params = sample_ppo_params(trial)
    # Initialize environment
    fourth_parties = list(network[network["Type"]=="Fourth-Party"]["ID"])
    env = CyberAttack(network,compromised_nodes=fourth_parties,shotgun_attack=False)
    env.reset()
    env = ActionMasker(env,mask_fn)
    # Sometimes, random hyperparameters can generate NaN
    try:
        model = MaskablePPO("MultiInputPolicy",env,verbose=0,**params)
        model.learn(total_timesteps=100000)      
    except (AssertionError,ValueError):
        raise optuna.TrialPruned()
    # Return average reward over 100 episodes
    rewards = []
    n_episodes = 0
    reward_sum = 0.0
    obs = env.state
    while n_episodes < 100:
        action_masks = mask_fn(env)
        action,states = model.predict(obs,action_masks=action_masks,deterministic=True)
        obs,reward,done,info = env.step(action)
        reward_sum += reward
        if done:
            rewards.append(reward_sum)
            reward_sum = 0.0
            n_episodes += 1
            env = CyberAttack(network,compromised_nodes=fourth_parties,shotgun_attack=False)
            env.reset()
            env = ActionMasker(env,mask_fn)
            obs = env.state
    final_reward = np.mean(rewards)
    return final_reward

study = optuna.create_study(study_name="optuna_ppo",
                            direction="maximize",
                            pruner=optuna.pruners.MedianPruner(),
                            load_if_exists=True)

for _ in range(100):
    try:
        study.optimize(objective,n_trials=1)
    except:
        continue
    
joblib.dump(study,"optuna_ppo.pkl")