# RL_CyberattackStrategies

With the number and severity of cyberattacks rapidly increasing, cybersecurity is a growing priority for companies around the world. In response to the mounting threat, many insurance companies and regulatory agencies have begun implementing machine learning models to estimate cybersecurity risks more reliably. Most of the models are either supervised in nature (trained on a dataset of normal versus harmful activity), or unsupervised in nature (clustering to find suspicious activity). Yet, there is very little research implementing reinforcement learning for cybersecurity problems.

Along with only considering supervised and unsupervised learning, traditional models have primarily focused on predicting risk by examining a company’s internal defenses and protocols such as firewalls and two-factor authentication. However, many recent high-profile events such as the infamous SolarWinds attack have demonstrated that external supply chains greatly magnify the effect of a single cyberattack across many companies. Given the lack of cybersecurity research in reinforcement learning and supply chains, this project proposes a theoretical reinforcement learning approach to predicting cyber risk in a supply chain.

## Methodology

We generate a synthetic network of nodes representing individual companies and edges representing supply chain connections. At each time step, an agent will be at a state defined as the collection of all known information for each node in the network. When at an individual node, an agent can take various actions representing attack strategies. The attack strategies have different probabilities of success and can target a different number of companies. The agent incurs a cost for each failed attack and a reward proportional the size of a newly breached company. Once the agent is trained, we can evaluate performance by simulating agent actions starting at random nodes in the network. If certain nodes are compromised more frequently, we can assume these are high-value targets. Describing the features of these high-value targets may be interesting and predictive from a cybersecurity perspective. For example, we may find an agent targets smaller but highly connected nodes instead of simply the largest companies. If the project is successful, one could apply the reinforcement learning approach to the global supply chain and discover highly vulnerable companies, potentially avoiding the next SolarWinds attack.

## Structure
Folder structure:
```
base_folder
│   Environment.py (Custom Gym environment)
│   Network.py (Synthetic supply chain network creation)
│   Empirical_Performance.ipynb (RL performance on small network with known theoretical performance)
│   README.md
└───Evaluation
│   │   Evaluation.ipynb (Visualizations of cyberattack risk)
│   │   Monte_Carlo.py (Run Monte Carlo evaluation)
│   │   Single_Run_Results (Monte Carlo sampling results go here)
└───Optuna
│   │   optuna_ppo.py (Run hyperparameter tuning)
│   │   optuna_ppo.pkl (Best hyperparameters)
│   │   Visualizations.ipynb (Visualizations of hyperparameter tuning)
└───Train
│   │   logs (Tensorboard training logs)
│   │   models (Models during training)
│   │   train.py (Run model)
```

## Training
First, you must generate a synthetic supply chain network using Network.py.
Next, you tune hyperparameters for the network using the optuna_ppo.py file and train a RL model using the train.py file.
Once the model has converged, you evaluate performance with Monte Carlo sampling using the Monte_Carlo.py file. Individual evaluation runs go in the Single_Run_Results folder. You can visualize the final cyberattack risk predicitons with the Evaluation.ipynb file.

## Configuration
All results were obtained by submitting cluster scripts to the [MIT SuperCloud](http://supercloud.mit.edu/).
Required libraries:
- anaconda/2022a
- glob
- gym
- joblib
- matplotlib
- networkx
- numpy
- optuna
- pandas
- sb3-contrib
- scipy
- seaborn
- stable_baselines3
- statsmodels
