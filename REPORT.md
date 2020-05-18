
# Project 1: Navigation

### Learning Algorithm
Deep Q-learning is an adaptation of the traditional Q-learning algorithm that employs a deep neural network in replacement of a Q-table. This deep neural network provides Q-values similar to a Q-table. Unlike the Q-table the deep neural network can handle more complex continuous state spaces. 
##### Reinforcement Learning
To give some background this learning algorithm is typically deployed in a Reinforcement Learning framework. This framework involves an agent, a set of states, a set of actions, and a reward component. An agent in this environment is given a state and selects an action based on this current state. Given the current state and the selected action, the environment then transitions to a new state. The agent is given the new state and a reward for taking the previous action in the previous state. This process continues indefinitely or until the agent reaches a terminal state. 
##### Q-Values and update function
Q-values correspond to the estimation of how good it is to take a particular action in a given state. The deep neural network predicts these values to choose the action an agent will take based on the current state. The rewards returned by the environment are used to help generate better estimates of these Q-values and are used to train the network and push it towards a better policy. 

![alt text](/images/q_update.png?raw=true "Q-learning update function")

At a high level the agent takes an action in a state and receives a new state and a reward. The agent then uses the state, action, reward, next state, and next predicted action to generate a better estimate of the Q-values in that previous state. The combination of the reward and the Q-value of the next predicted action in the next state is a similar estimation to that of the Q-value of the previously take action in that previous state. This however is a better estimation because this estimation also relies on the reward which is the actual value of how good that previous action. The error between this new estimation and the old estimation can be used to slightly push the network towards a better policy. 
##### Experience Replay

##### Hyperparamters
| Name          | Value   | Description                               |
|---------------|---------|-------------------------------------------|
| BUFFER_SIZE   | 100,000 | Replay buffer size                        |
| BATCH_SIZE    | 64      | Minibatch size                            |
| GAMMA         | 0.99    | Discount factor                           |
| TAU           | 0.001   | Soft update of target paramters           |
| LR            | 0.0005  | Learning rate                             |
| UPDATE_EVERY  | 4       | How often to update network (steps)       |
| A             | 0.6     | Replay buffer prioritization weight       |
| B             | 0.4     | Importance sampling weight scaling factor |
| B_GROWTH_RATE | 0.005   | Growth rate of B over time until B=1      |
| EPS_START     | 1       | Start value of epsilon                    |
| EPS_END       | 0.01    | End value of epsilon                      |
| EPS_DECAY     | 0.995   | Decay of epsilon until epsilon=EPS_END    |
| N_EPISODES    | 2000    | Max number of episodes in learning loop   |
##### Network Architecture

![alt text](/images/model_architecture.png?raw=true "Neural Network Architecture")
