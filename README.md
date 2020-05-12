# Project 1: Navigation

### Project Details

This project involves training an agent to navigate through a large square environment filled with bananas. The agent's goal is to collect as many yellow bananas as possible while also avoiding the blue bananas. The agent receives a reward of +1 for collecting yellow bananas and a reward of -1 for collecting blue bananas. 

##### State Space
The state space available to the agent includes 37 dimensions. These dimensions include the agents velocity as well as ray-based perception of objects around the agent's forward direction. 
##### Action Space
There are four discrete actions available to the agent which are 0, 1, 2, 3. These actions correspond to move forward, move backward, turn left, and turn right.
##### Solving Criteria
This particular environment is considered solved when the agent is able to achieve an average score of +13 over 100 consecutive episodes. 

### Getting Started
To set up your python environment to run the code in this repository, follow the instructions below.

1.  Create (and activate) a new environment with Python 3.6.
    
-   **Linux**  or  **Mac**:
    
```
    conda create --name drlnd python=3.6
    source activate drlnd
```
   -   **Windows**:
  
```  
    conda create --name drlnd python=3.6
    activate drlnd
```
2.  Follow the instructions in  [this repository](https://github.com/openai/gym)  to perform a minimal install of OpenAI gym.
    
    -   Next, install the  **classic control**  environment group by following the instructions  [here](https://github.com/openai/gym#classic-control).
    -   Then, install the  **box2d**  environment group by following the instructions  [here](https://github.com/openai/gym#box2d).
3.  Next install the packages listed in requirements.txt using the pip install command
```
pip install .
```
4.  Create an  [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html)  for the  `drlnd`  environment.

python -m ipykernel install --user --name drlnd --display-name "drlnd"

5.  Before running code in a notebook, change the kernel to match the  `drlnd`  environment by using the drop-down  `Kernel`  menu.
### Instructions
To train the agent simply start the Jupyter notebook and follow the instructions listed in the cells. 
