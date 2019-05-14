import numpy as np
import copy
import os
import pandas as pd

class Simulator():
    def __init__(self,**kwargs):
        # initialize containers for grid, hazard locations, agent and goal locations, etc.,
        self.grid = []
        self.hazards = []
        self.agent = []
        self.goal = []
        
        # initialize global variables e.g., height and width of gridworld, hazard penalty value
        self.width = 0
        self.height = 0
        self.world_size = ''
        self.world_type = ''
        
        # setup standard reward value
        self.standard_reward = -0.001            
        self.hazard_reward = 10000*self.standard_reward          
        self.goal_reward = 0 
            
        # setup world
        world_name = ''
        if "world_size" not in kwargs:
            print ('world_size parameter required, choose either small or large')
            return
        
        if "world_type" not in kwargs:
            print ('world_type parameter required, choose maze, random, or moat')

        ### set world size ###    
        if kwargs["world_size"] == 'small':
            self.world_size = 'small'
            self.width = 13
            self.height = 11

        if kwargs["world_size"] == 'large':
            self.world_size = 'large'
            self.width = 41
            self.height = 15

        ### initialize grid based on world_size ###
        self.grid = np.zeros((self.height,self.width))
        self.max_steps_per_episode = 2*self.width*self.height
        
        # index states for Q matrix
        self.states = []
        for i in range(self.height):
            for j in range(self.width):
                block = tuple([i,j])
                self.states.append(block)
                
        ### with world type load in hazards ###
        if kwargs["world_type"] == 'maze':
            self.world_type = 'maze'
            self.agent = tuple([self.height-2, 0])               # initial location agent
            self.goal = tuple([self.height-2, self.width-1])     # goal block   
            
        if kwargs["world_type"] == 'maze_v2':
            self.world_type = 'maze_v2'
            self.agent = tuple([self.height-2, 0])        
            self.goal = tuple([self.height-2, self.width-1])       

        if kwargs["world_type"] == 'random':
            self.world_type = 'random'
            self.agent = tuple([0,0])   
            self.goal = tuple([0,self.width-1])    

        if kwargs["world_type"] == 'moat':
            self.world_type = 'moat'
            self.agent = tuple([0,0])   
            self.goal = tuple([0,self.width-1])   

        ### load in hazards for given world size and type ###  
        location = os.path.dirname(os.path.realpath(__file__))

        hazard_csvname = location + '/gridworld_levels/' + kwargs["world_size"] + '_' + kwargs["world_type"] + '_hazards.csv'
        
        # load in preset hazard locations from csv
        self.hazards = pd.read_csv(hazard_csvname,header = None)
            
        # initialize hazards locations
        temp = []
        for i in range(len(self.hazards)):
            block = list(self.hazards.iloc[i])
            self.grid[block[0]][block[1]] = 1   
            temp.append(block)
                
        # initialize hazards location
        self.hazards = temp
        self.hazards = [tuple(v) for v in self.hazards]

        ### initialize state index, Q matrix, and action choices ###
        # initialize action choices
        self.action_choices = {'down': [-1,0],'up': [1,0],'left':[0,-1],'right':[0,1]}
         
        # other params
        self.max_steps = 2*self.width*self.height  # maximum number of steps per episode
        self.num_states = self.height*self.width
        self.state_range = [np.min(np.array(self.states),0),np.max(np.array(self.states),0)]
        self.num_actions = 4
        
        if 'max_steps' in kwargs:
            self.max_steps = kwargs['max_steps']
            self.max_steps_per_episode = kwargs['max_steps']
            
                                    
    ################## f_system rules ##################
    # state initializer - note this is in index form
    def reset(self):
        self.state = tuple([np.random.permutation(self.height)[0],np.random.permutation(self.width)[0]])
        self.current_step = 0
        return self.state
    
    ### reward rule ###
    def get_reward(self,state_tuple):
        reward = 0
        # if new state is goal set reward of 0
        if state_tuple == self.goal:
            reward = self.goal_reward
        elif state_tuple in self.hazards:
            reward = self.hazard_reward
        else:  # standard non-hazard square
            reward = self.standard_reward
        return reward          

    ### main f_system function ###
    def step(self,action):
        # update step
        self.current_step += 1
        
        # convert action to movement
        action_tuple = self.action_choices[action]
        
        # move to new state
        new_state_tuple = [a + b for a,b in zip(self.state,action_tuple)]
        
        # don't go anywhere if movement takes you off gridworld
        if new_state_tuple[0] > self.height-1 or new_state_tuple[0] < 0 or new_state_tuple[1] > self.width-1 or new_state_tuple[1] < 0:  
            new_state_tuple = self.state
        else:
            new_state_tuple = tuple(new_state_tuple)
        reward = self.get_reward(new_state_tuple)
        
        # update state 
        self.state = new_state_tuple
        
        # has the agent reached goal?  Have you run out of iterations?
        done = False
        if self.current_step == self.max_steps or self.state == self.goal:
            done = True
        
        # return
        return self.state, reward, done