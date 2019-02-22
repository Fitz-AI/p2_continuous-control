
# Tony Quertier - Project 2 Continuous control


## Examine the State and Action Spaces

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


##  Train the agent


### Algorithm

To train the agent, we use the ddpg algorithm form DDPG_continuous_control notebook. 
DDPG (Lillicrap, et al., 2015), short for Deep Deterministic Policy Gradient, is a model-free off-policy actor-critic algorithm, combining DPG with DQN.The original DQN works in discrete space, and DDPG extends it to continuous space with the actor-critic framework while learning a deterministic policy.

In order to do better exploration, an exploration policy μ’ is constructed by adding noise N :μ′(s)=μθ(s)+N.

In addition, DDPG does soft update on the parameters of both actor and critic, with τ≪1: θ′←τθ+(1−τ)θ′. In this way, the target network values are constrained to change slowly, different from the design in DQN that the target network stays frozen for some period of time.

### Hyper parameters for V1

BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 128        
GAMMA = 0.97            
TAU = 1e-3              
LR_ACTOR = 1e-4         
LR_CRITIC = 1e-4        
WEIGHT_DECAY = 0.0   

### Hyper parameters for V2

BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 64        
GAMMA = 0.99           
TAU = 1e-3              
LR_ACTOR = 1e-3         
LR_CRITIC = 1e-3        
WEIGHT_DECAY = 0.0  

### Neural Networks for V1

Actor and Critic network models were defined in model.py.

The Actor networks utilised two fully connected layers with 256 and 128 units with relu activation and tanh activation for the action space. 
The Critic networks utilised two fully connected layers with 256 and 128 units with relu activation. 

### Neural Networks for V2

Actor and Critic network models were defined in model.py.

The Actor networks utilised two fully connected layers with 200 and 400 units with relu activation and tanh activation for the action space. 
The Critic networks utilised two fully connected layers with 200 and 400 units with relu activation. 

### Number of agents

In Continuous_control_V1 we use only one agent and in Continuous_Control_V2 we use 20 agents. We can see below the difference of performances between the two methods.



## Performance of the agent

For V1, we have :

![Alt text](https://github.com/Quertier/p2_continuous-control/blob/master/p2_continuous_control_V1.PNG)

For V2, we have :

![Alt text](https://github.com/Quertier/p2_continuous-control/blob/master/p2_continuous_control_V2.PNG)


## Future Improvements

As future works, we could explore Proximal Policy Optimization (PPO) and Distributed Distributional Deterministic Policy Gradients (D4PG) methods.




