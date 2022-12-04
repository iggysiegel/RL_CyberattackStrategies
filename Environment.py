import numpy as np
import gym
from gym.spaces import Dict, MultiBinary, Box, Discrete

class CyberAttack(gym.Env):
    
    """
    A simulated supply chain cyberattack environment.
    """
    
    def __init__(self,network,compromised_nodes=[],step_limit=1000,shotgun_attack=True):
        """
        Initialize CyberAttack environment.
        
        Inputs:
            network : dataframe - supply chain network
            compromised_nodes : list - IDs of potential starting compromised nodes
            step_limit : int - maximum number of steps per episode
            shotgun_attack : bool - whether or not shotgun attack is available
            
        The supply chain network should have columns ID, Type, Size, Incoming_Connections,
        Outgoing_Connections, and All_Connections. The ID of each node should be consecutive
        integers starting at zero. Use class Network to create a suitable synthetic supply chain network.
        """
        self.dtype_int = np.int32
        self.dtype_float = np.float32
        
        self.step_current = 0
        self.step_limit = step_limit
        
        self.network = network
        self.n = len(network)

        self.starting_compomised_nodes = compromised_nodes

        self.observation_space = Dict({"current_node": Discrete(self.n),
                                       "compromised": MultiBinary(self.n),
                                       "accessible_out": MultiBinary(self.n),
                                       "accessible_in": MultiBinary(self.n),
                                       "size": Box(low=0,
                                                   high=network["Size"].max(),
                                                   shape=(self.n,),
                                                   dtype=self.dtype_int)})
        
        if shotgun_attack:
            self.action_space = Discrete(self.n+1)
            self.shotgun_attack = self.n
        else:
            self.action_space = Discrete(self.n)
            self.shotgun_attack = None
    
    def reset(self):
        """
        Reset the environment to the initial state defined by the supply chain network.
        
        Outputs:
            self.state : initial state of environment
        """ 
        self.step_current = 0

        if self.starting_compomised_nodes == []:
            self.compromised_node = np.random.randint(self.n)
        else:
            self.compromised_node = np.random.choice(self.starting_compomised_nodes)
        
        compromised_list = [1 if i==self.compromised_node else 0 for i in range(self.n)]
        accessible_out_list = self.network.iloc[self.compromised_node]["Outgoing_Connections"]
        accessible_out_list = [1 if i in accessible_out_list else 0 for i in range(self.n)]
        accessible_in_list = self.network.iloc[self.compromised_node]["Incoming_Connections"]
        accessible_in_list = [1 if i in accessible_in_list else 0 for i in range(self.n)]
        
        self.state = {}
        self.state["current_node"] = self.compromised_node
        self.state["compromised"] = np.array(compromised_list,dtype=self.dtype_int)
        self.state["accessible_out"] = np.array(accessible_out_list,dtype=self.dtype_int)
        self.state["accessible_in"] = np.array(accessible_in_list,dtype=self.dtype_int)
        self.state["size"] = np.array(list(self.network["Size"]),dtype=self.dtype_int)
        
        return self.state
        
    def step(self,action):
        """
        Run one step of the environment using action.
        
        Inputs:
            action : int - ID of node to attack.
        Outputs:
            self.state : resulting state of environment from performing action
            reward : float - reward from performing action
            done : bool - whether or not episode has terminated
            {} : dict - auxiliary information regarding step
        """
        # Case when shotgun attack
        if action == self.shotgun_attack:
            # For every supply chain connection
            targets = self.network.iloc[self.state["current_node"]]["All_Connections"]
            targets = [i for i in targets if self.state["compromised"][i] != 1]
            shotgun_reward = 0
            while len(targets) > 0:
                target = targets.pop()
                # Incoming edge attack
                if self.state["accessible_in"][target]==1:
                    p = 0.1 * min(.2,self.state["size"][target]**(-.25))
                # Outgoing edge attack
                else:
                    p = 0.1 * 0.1 * min(.2,self.state["size"][target]**(-.25))
                # If unsuccessful
                if np.random.rand() > p:
                    continue
                # If sucessful
                else:
                    self.state["current_node"] = target
                    self.state["compromised"][target] = 1
                    accessible_out_list = []
                    new_accessible_out_list = self.network.iloc[target]["Outgoing_Connections"]
                    accessible_in_list = []
                    new_accessible_in_list = self.network.iloc[target]["Incoming_Connections"]
                    for i in range(self.n):
                        if self.state["accessible_out"][i]==1 or i in new_accessible_out_list:
                            accessible_out_list.append(1)
                        else:
                            accessible_out_list.append(0)
                        if self.state["accessible_in"][i]==1 or i in new_accessible_in_list:
                            accessible_in_list.append(1)
                        else:
                            accessible_in_list.append(0)
                    self.state["accessible_out"] = np.array(accessible_out_list,dtype=self.dtype_int)
                    self.state["accessible_in"] = np.array(accessible_in_list,dtype=self.dtype_int)
                    shotgun_reward += self.state["size"][target]/1000
            # Total shotgun attack reward
            if shotgun_reward > 0:
                reward = shotgun_reward
            else:
                reward = -1
        # Case when node is already compromised
        elif self.state["compromised"][action]==1:
            reward = -1
        # Case when node is not accessible
        elif self.state["accessible_in"][action]==0 and self.state["accessible_out"][action]==0:
            reward = -1
        # Case when directed attack
        else:
            # Incoming edge attack
            if self.state["accessible_in"][action]==1:
                p = min(.2,self.state["size"][action]**(-.25))
            # Outgoing edge attack
            else:
                p = 0.1 * min(.2,self.state["size"][action]**(-.25))
            # If unsuccessful
            if np.random.rand() > p:
                reward = -1
            # If successful
            else:
                self.state["current_node"] = action
                self.state["compromised"][action] = 1
                accessible_out_list = []
                new_accessible_out_list = self.network.iloc[action]["Outgoing_Connections"]
                accessible_in_list = []
                new_accessible_in_list = self.network.iloc[action]["Incoming_Connections"]
                for i in range(self.n):
                    if self.state["accessible_out"][i]==1 or i in new_accessible_out_list:
                        accessible_out_list.append(1)
                    else:
                        accessible_out_list.append(0)
                    if self.state["accessible_in"][i]==1 or i in new_accessible_in_list:
                        accessible_in_list.append(1)
                    else:
                        accessible_in_list.append(0)
                self.state["accessible_out"] = np.array(accessible_out_list,dtype=self.dtype_int)
                self.state["accessible_in"] = np.array(accessible_in_list,dtype=self.dtype_int)
                reward = self.state["size"][action]/1000
        # Check if episode finished
        self.step_current += 1
        if self.step_current >= self.step_limit or self.state["compromised"].sum() == self.n:
            done = True
        else:
            done = False
        # Return output
        return self.state,reward,done,{}
    
    def render_action(self,action):
        """
        Render observation
        """
        number_compromised_nodes = self.state["compromised"].sum()
        print(f"Time: {self.step_current} -> Number Compromised Nodes: {number_compromised_nodes}")
        
    def render_action(self,action):
        """
        Render observation with action
        """
        number_compromised_nodes = self.state["compromised"].sum()
        if action != self.shotgun_attack:
            print(f"Time: {self.step_current} -> Action: {action} -> Number Compromised Nodes: {number_compromised_nodes}")
        else:
            print(f"Time: {self.step_current} -> Shotgun Attack -> Number Compromised Nodes: {number_compromised_nodes}")