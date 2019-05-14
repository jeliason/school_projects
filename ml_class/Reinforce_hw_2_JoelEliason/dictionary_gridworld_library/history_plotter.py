import matplotlib.pyplot as plt
import numpy as np

def plot_cost_history(episode_rewards):
    # create figure
    fig = plt.figure(figsize = (12,5))
    ax = fig.add_subplot(1,1,1)

    # plot total reward history
    episode_rewards = np.array(episode_rewards)
    ax.plot(episode_rewards.flatten())
    ax.set_xlabel('episode')
    ax.set_ylabel('total episode reward')
    plt.show()