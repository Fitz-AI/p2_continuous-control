
# Tony Quertier - Project 1 : Navigation

---


## Examine the State and Action Spaces

The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 

The goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas. In order to solve the environment, our agent must achieve an average score of +13 over 100 consecutive episodes.

##  Train the agent

As reinforcement learning, I started with a Deep Q-Network agent. To improve the algorithm, I implemented Double DQN algorithm to handle the problem of overestimation of Q_values and Dueling DQN that can learn which states are valuable or not without having to learn the effect of each actions at each state.

### Dueling DQN

The algorithm Dueling DQN can learn which states are valuable without having to learn the effect of each action at each state. If we set V(s) the state value at state s and A(s,a) the advantage of taking action a at state s then we have:

![Alt text](https://github.com/Quertier/p1-navigation/blob/master/Images/Dueling_DQN.png)


```python
class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, action_size)
        self.fc_val = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x_adv = self.fc3(x)
        x_val = self.fc_val(x)
        return x_adv + x_val
```

### Double DQN

The aim of Double DQN is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation.

![Alt text](https://github.com/Quertier/p1-navigation/blob/master/Images/Double_DQN.png)


```python
def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        #For DQN
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #Q_expected = self.qnetwork_local(states).gather(1, actions)

        # For Double DQN : Q(s,a) = r(s,a) + gamma*Q(s',argmax_aQ(s',a))
        #argmax_aQ(s',a)
        max_actions = self.qnetwork_local(next_states).detach().argmax(dim=1, keepdim=True)
        #Q(s',argmax_aQ(s',a))
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(dim=1, index=max_actions)
        #Q(s,a)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(dim=1, index=actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  
```

### Hyperparameters


```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.999            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```

## Performance of the agent

We set eps_start=1.0, eps_end=0.01 and eps_decay=0.99. We have:

![Alt text](https://github.com/Quertier/p1-navigation/blob/master/Images/p1-navigation.PNG)

If you want to test it by yourself :) 


```python
from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
%matplotlib inline
from Dqn_agent import Agent
```


```python
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
```

    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		
    Unity brain name: BananaBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 37
            Number of stacked Vector Observation: 1
            Vector Action space type: discrete
            Vector Action space size (per agent): 4
            Vector Action descriptions: , , , 
    


```python
brain_name = env.brain_names[0] # get the name of the brains from the Unity environment
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name] # reset the environment and obtain info on state/action space

# initialize agent with state size and action size.
agent = Agent(len(env_info.vector_observations[0]), brain.vector_action_space_size, seed=0)

# load the trained weights
agent.qnetwork_local.load_state_dict(torch.load('Dueling_model.pth'))

state = env_info.vector_observations[0]  # get the first state
score = 0 # initialize the score
while True: # loop until the episode ends
    action = agent.act(state, 0).astype(np.int32) # select a greedy action
    env_info = env.step(action)[brain_name] # take that action
    score += env_info.rewards[0] # update the score with the reward for taking that action
    next_state = env_info.vector_observations[0] # the next state
    state = next_state # set current state to next state
    done = env_info.local_done[0] # get the value of the done bool, indicating the episode is over
    # end episode if done is true
    if done:
        break

print("Score: {}".format(score)) # print the score


```

    Score: 19.0
    

## Future Improvements

To improve the agent, I have to implement prioritized experience replay. The idea is to decide which experiences could be more important than others for the training.
