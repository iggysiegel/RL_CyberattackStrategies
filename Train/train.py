import sys
sys.path.append('../')
from Network import Network
from Environment import CyberAttack
import os
import numpy as np
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

### Create supply-chain network
network = Network(1000,random_state=0).network
fourth_parties = list(network[network["Type"]=="Fourth-Party"]["ID"])

### Create cyber attack environment
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

env = CyberAttack(network,compromised_nodes=[],shotgun_attack=0)
env.reset()
env = ActionMasker(env,mask_fn)

### Logging and saving
models_dir = "models/PPO"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

### Train model
params = {'learning_rate': 0.00016039989215808857,
          'n_steps': 2048,
          'batch_size': 1024,
          'gamma': 0.999,
          'clip_range': 0.4}
          
model = MaskablePPO("MultiInputPolicy",
                    env,
                    verbose=0,
                    **params,
                    tensorboard_log=logdir)

TIMESTEPS = 100000

i = 1
while True:
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1