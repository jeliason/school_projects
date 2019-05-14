# import basic utilities
import numpy as np
import copy

# import numpy and matplotlib utilities
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from IPython.display import clear_output
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

class Visualizer():
    def __init__(self,simulator):
        # extract values for plotting
        self.grid = simulator.grid
        self.goal = simulator.goal
        self.agent = simulator.agent
        self.height = simulator.height
        self.width = simulator.width
        self.states = simulator.states
        self.actions = simulator.action_choices
        self.simulator = simulator
        
    ### baic color ###
    def color_gridworld(self):
        # define colors
        colors = [(0.9,0.9,0.9),(255/float(255), 119/float(255), 119/float(255)), (66/float(255),244/float(255),131/float(255)), (1/float(255),100/float(255),200/float(255)),(0,0,0)]   
        self.my_cmap = LinearSegmentedColormap.from_list('colormapX', colors, N=100)
        
    ### world coloring function ###
    def show_gridworld(self,**kwargs):
        # color gridworld
        self.color_gridworld()
        
        # copy grid for plotting, add agent and goal location
        p_grid = copy.deepcopy(self.grid)
        p_grid[self.goal[0]][self.goal[1]] = 2   
        p_grid[self.agent[0]][self.agent[1]] = 3   
        
        # check if lights off
        if 'lights' in kwargs:
            # if lights off color every square black except current square and adjacent squares that can be 'seen' by the agent
            if kwargs['lights'] == 'off':
                for i in range(self.height):
                    for j in range(self.width):
                        if np.abs(i - self.agent[0]) + np.abs(j - self.agent[1]) > 1:
                            p_grid[i][j] = 4
                            
        # plot gridworld
        ax = 0
        if 'ax' in kwargs:
            ax = kwargs['ax']
        else: 
            fsize = 6
            if self.width > 20:
                fsize = 16
            fig = plt.figure(figsize = (fsize,6),frameon=False)
            ax = fig.add_subplot(111, aspect='equal')

        ax.pcolormesh(p_grid,edgecolors = 'k',linewidth = 0.01,vmin=0,vmax=4,cmap = self.my_cmap)

        # clean up plot
        ax.axis('off')
        ax.set_xlim(-0.1,self.width);
        ax.set_ylim(-0.1,self.height);
        plt.show()

    ############### draw arrow map after training ###############
    ### setup arrows
    def add_arrows(self,ax,state,action,scale,arrow_length):
        x = state[1]
        y = state[0]
        dx = 0
        dy = 0

        ### switch for starting point of arrow depending on action - so that arrow always centered ###
        if action == 'down':    # action == down
            y += 0.9
            x += 0.5
            dy = -0.8
        if action == 'up':      # action == up
            x += 0.5
            y += 0.1
            dy = 0.8
        if action == 'left':    # action == left
            y += 0.5
            x += 0.9
            dx = -0.8
        if action == 'right':   # action == right
            y += 0.5
            x += 0.1
            dx = 0.8

        ### add patch with location / orientation determined by action ###
        ax.add_patch(
           patches.FancyArrowPatch(
           (x, y),
           (x+dx, y+dy),
           arrowstyle='->',
           mutation_scale=scale,   # 30 for small map, 20 for large
           lw=arrow_length,              # 2 for small map, 1.5 for large
           color = 'k',
           )
        )

    # best action map
    def draw_arrow_map(self,Q):  
        ### compute optimal directions ###
        num_states = len(self.states)
        q_dir = []
        for state in self.simulator.states:
            action = max(Q[state], key=Q[state].get)
            q_dir.append(action)

        ### plot arrow map ###
        colors = [(0.9,0.9,0.9),(255/float(255), 119/float(255), 119/float(255)), (66/float(255),244/float(255),131/float(255)), (1/float(255),100/float(255),200/float(255)),(0,0,0)]  
        my_cmap = LinearSegmentedColormap.from_list('colormapX', colors, N=100)

        ### setup grid
        p_grid = self.simulator.grid
        p_grid[self.simulator.goal[0]][self.simulator.goal[1]] = 2   
        
        ### setup variables for figure and arrow sizes ###
        fsize = 6
        scale = 30                    # 30 for small map, 20 for large
        arrow_length = 2              # 2 for small map, 1.5 for large
        if np.shape(p_grid)[0] > 14 or np.shape(p_grid)[1] > 14:
            fsize = 16
            scale = 20
            arrow_length = 1.5

        # setup figure if not provided
        fig = plt.figure(figsize = (fsize,6),frameon=False)
        ax = fig.add_subplot(111, aspect='equal')
            
        # plot regression surface 
        ax.pcolormesh(p_grid,edgecolors = 'k',linewidth = 0.01,vmin=0,vmax=4,cmap = my_cmap)

        # clean up plot
        ax.axis('off')
        ax.set_xlim(-0.1,self.simulator.width);
        ax.set_ylim(-0.1,self.simulator.height); 

        ### go over states and draw arrows indicating best action
        # switch size of arros based on size of gridworld (for visualization purposes)
        for i in range(len(self.states)):
            state = self.states[i]
            if state != self.simulator.goal:  
                action = q_dir[i]
                self.add_arrows(ax,state,action,scale,arrow_length)
        plt.show()
          
    # plot reward history
    def plot_reward_history(self,episode_rewards):
        # create figure
        fig = plt.figure(figsize = (12,5))
        ax = fig.add_subplot(1,1,1)

        # plot total reward history
        episode_rewards = np.array(episode_rewards)
        ax.plot(episode_rewards.flatten())
        ax.set_xlabel('episode')
        ax.set_ylabel('total episode reward')
        plt.show()