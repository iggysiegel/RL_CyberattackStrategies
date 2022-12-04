import sys
sys.path.append('../')
from Network import Network
from Environment import CyberAttack 
import numpy as np
import glob
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

### Create supply-chain network
network = Network(1000,random_state=0).network

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

env = CyberAttack(network,compromised_nodes=[],shotgun_attack=False)
env.reset()
env = ActionMasker(env,mask_fn)

### Load model
models_dir = "../Train/models/PPO"
model_path = f"{models_dir}/19300000"
model = MaskablePPO.load(model_path,env)

### Run simulation
results = np.zeros(len(network))
for i in range(1000):
    env.reset()
    obs = env.state
    for j in range(1000):
        action_masks = mask_fn(env)
        action,states = model.predict(obs,action_masks=action_masks,deterministic=True)
        obs,reward,done,info = env.step(action)
    results += env.state["compromised"]
    
### Save results
num_files = len(glob.glob("Single_Run_Results/*.npy"))
np.save(f"Single_Run_Results/results_{num_files}.npy",results)