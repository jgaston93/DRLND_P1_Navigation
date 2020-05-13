
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

3. Follow the instructions [here](https://pytorch.org/)  to install pytorch.

5.  Next enter the python directory and install the packages listed in requirements.txt using the pip install command
```
cd python
pip install .
```
6.  Create an  [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html)  for the  `drlnd`  environment.
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
7.  Before running code in a notebook, change the kernel to match the  `drlnd`  environment by using the drop-down  `Kernel`  menu.

8. Finally you need to download the pre-built unity training environment and unzip this into the repo
	-   Linux:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
	-   Mac OSX:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
	-   Windows (32-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
	-   Windows (64-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
### Instructions
To train the agent simply start the Jupyter notebook and step through `Navigation.ipynb`
