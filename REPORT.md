
# Project 1: Navigation

### Learning Algorithm
Deep Q-learning is an adaptation of the traditional Q-learning algorithm that employs a deep neural network in replacement of a Q-table. This deep neural network provides Q-values similar to a Q-table. Unlike the Q-table the deep neural network can handle more complex continuous state spaces. 
##### Reinforcement Learning

![Reinforcement Learning](/images/reinforcement_learning.png?raw=true "Reinforcement Learning")

To give some background this learning algorithm is typically deployed in a Reinforcement Learning framework. This framework involves an agent, a set of states, a set of actions, and a reward component. An agent in this environment is given a state and selects an action based on this current state. Given the current state and the selected action, the environment then transitions to a new state. The agent is given the new state and a reward for taking the previous action in the previous state. This process continues indefinitely or until the agent reaches a terminal state. 
##### Q-Values and update function

![Q-function](/images/q_function.png?raw=true "Q-function")

Q-values correspond to the estimation of how good it is to take a particular action in a given state. The deep neural network predicts these values to choose the action an agent will take based on the current state. The rewards returned by the environment are used to help generate better estimates of these Q-values and are used to train the network and push it towards a better policy. 

![Q-learning update function](/images/q_update.png?raw=true "Q-learning update function")

At a high level the agent takes an action in a state and receives a new state and a reward. The agent then uses the state, action, reward, next state, and next predicted action to generate a better estimate of the Q-values in that previous state. The combination of the reward and the Q-value of the next predicted action in the next state is a similar estimation to that of the Q-value of the previously take action in that previous state. This however is a better estimation because this estimation also relies on the reward which is the actual value of how good that previous action. The error between this new estimation and the old estimation can be used to slightly push the network towards a better policy. 
##### Experience Replay

One other piece of background information that is useful in explaining the learning algorithm is the use of the experience replay buffer. As the agent is interacting with the environment it is useful to gather up past experiences. In the future the agent can pull these experiences back up and learn from them.

This is helpful in two ways. One problem that the replay buffer addresses is the problem of rarely visited state. All states will not be visited with equal probability and so saving these rare states gives the agent the ability to pull these back out and continue to learn from them. Another problem the replay buffer addresses is the correlation of states and time. The ability to randomly sample states from this buffer allows the agent to learn from many states that aren't next to each other in the temporal dimension. This helps build a more robust policy that is able to better generalize to new situations. 

##### Learning Loop

![Pseudocode](/images/pseudocode.png?raw=true "Pseudocode")

To put all of this together the agent first chooses an action based on the current state of the environment. The agent executes this action and observes the reward and the next state. The agent then stores the state, action, reward and next state into the experience replay buffer. Then once the buffer has reach a certain capacity the agent can begin sampling and learning from the buffer experiences. The agent samples a minibatch from the buffer and calculates the td-error based on this batch. Using this error the agent can update its policy using gradient descent. 

##### Other improvements

Along with experience replay this implementation all makes use of the other improvements listed in the course. These improvements include the Double DQN, Prioritized Experience Replay, and the Dueling DQN.

Double DQN involves having two neural networks. A second neural network is used to predict the td-target that is used to calculate the td-error. To calculate the td-target an action is chosen based on the online policy but the Q-value of this action is calculated based on the second offline network. This helps with the problem of overestimation of Q-values that occured in the vanilla update function. 

[Double DQN paper](https://arxiv.org/abs/1509.06461)

Prioritized Experience Replay is the process of ranking the experiences in the replay buffer. Experiences are ranked by their td-error such that experiences with larger td-errors will be sampled more frequently. The thought process behind this is that experiences with larger errors will provide more information for the agent to learn from.

[Prioritized Experience Replay paper](https://arxiv.org/abs/1511.05952)

Finally the Dueling DQN improvement is a modification of the neural network itself. This involves splitting the end of the network into two stream such that the network is predicting state values and advantage values. This network predicts the state value and a state-dependent advantage value for each of the actions. The advantage values are added to the state value to recover the final Q-values returned by the network. 

[Dueling DQN paper](https://arxiv.org/abs/1511.06581)


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
| B_START       | 0.4     | Importance sampling weight scaling factor |
| B_GROWTH_RATE | 0.00003 | Growth rate of B over time until B=1      |
| EPS_START     | 1       | Start value of epsilon                    |
| EPS_END       | 0.01    | End value of epsilon                      |
| EPS_DECAY     | 0.995   | Decay of epsilon until epsilon=EPS_END    |
| N_EPISODES    | 2000    | Max number of episodes in learning loop   |
##### Network Architecture

![Neural Network Architecture](/images/model_architecture.png?raw=true "Neural Network Architecture")

The network architecture implemented in this repo is shown by the figure above. The network takes an input vector with 37 dimenions. There are then two hidden layers with 128 neurons each that use a RELU activation function. 

Since this network is the Dueling Q-Network variation the output from the second hidden layer is split off into two separate layers. One of which contains four neurons and this represents the advantage values and the other layer contains one neuron which represents the state value. Both of these layers use linear activation functions and the outputs from these layers are summed together to produce a final output of 4 dimensions representing the Q-values. 

In total this network contains 21,760 tunable weights. 

### Plot of Rewards

```
Episode 100	Average Score: 1.03
Episode 200	Average Score: 3.90
Episode 300	Average Score: 7.32
Episode 400	Average Score: 11.15
Episode 500	Average Score: 12.01
Episode 557	Average Score: 13.04
Environment solved in 457 episodes!	Average Score: 13.04
```

![Reward Plot](/images/plot.png?raw=true "Reward Plot")

In this implementation the agent was able to solve the environment in 457 episodes. The above plot shows the scores in blue as the agent progresses through time. The importance sampling weight scaling factor, B, and Epsilon are plotted alongside the scores to show how these values change as the agent is converging to a sufficient policy. B is shown in red and Epsilon is shown in green. 

These plots could be helpful in showing how B and Epsilon affect convergence to better tune these two variables. 

##### Trained Agent

![Trained Agent](/images/trained_agent.gif?raw=true "Trained Agent")

A gif of the trained agent using the saved weights is seen above. The agent can navigate between blue bananas and specifically targets the yellow bananas. 

### Ideas for Future Work

