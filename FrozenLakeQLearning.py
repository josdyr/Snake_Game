import gym
import random
import numpy as np

import time
import blessings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gym.envs.registration import register
from IPython.display import clear_output
from blessings import Terminal


try:
    register(
        id='FrozenLakeNoSlip-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78,  # optimum = .8196
    )
except:
    pass


# env_name = 'FrozenLake-v0'
env_name = 'FrozenLakeNoSlip-v0'
env = gym.make(env_name)
print("Observation Space: ", env.observation_space)
print("Action Space: ", env.action_space)
type(env.action_space)


class Agent():
    def __init__(self, env):
        self.is_discrete = \
            type(env.action_space) == gym.spaces.discrete.Discrete

        if self.is_discrete:
            self.action_size = env.action_space.n
            print("Action size: ", self.action_size)
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range: ", self.action_low, self.action_high)

    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low,
                                       self.action_high,
                                       self.action_shape)
        return action


class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        self.state_size = env.observation_space.n
        print("State Size:", self.state_size)

        self.total_reward_plot = []
        self.episode_plot = []
        self.steps_plot = []

        self.epsilon = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        self.q_table = 0.0001 * np.random.random([self.state_size, self.action_size])

    def get_action(self, state):
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.epsilon else action_greedy

    def train(self, experience):
        state, action, next_state, reward, done = experience

        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount_rate * np.max(q_next) # equation

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update

        if done:
            self.epsilon = self.epsilon * 0.99


def plot_seaborn(episode_plot, total_reward_plot, steps_plot):
    # sns.set(color_codes=True)
    # ax = sns.regplot(np.array([episode_plot])[0], np.array([total_reward_plot])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    # ax.set(xlabel='episodes', ylabel='total_reward')
    # new_array_counter = [x + 1 for x in episode_plot]
    import ipdb; ipdb.set_trace()
    plt.plot(episode_plot, total_reward_plot, episode_plot, steps_plot, )
    # plt.plot(episode_plot, total_reward_plot)
    plt.savefig('basic_figure.png')
    plt.show()


t = Terminal()
agent = QAgent(env)

total_reward = 0
for episode in range(300): # 1500 10000
    state = env.reset()
    done = False
    steps = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train((state, action, next_state, reward, done))

        state = next_state
        total_reward += reward

        print("reward:", reward, "state:", state, "action:", action, "episode:", episode, "steps:", steps, "total_reward:", total_reward, "epsilon:", agent.epsilon)
        # env.render()

        # print(agent.q_table)

        # time.sleep(.01)
        clear_output(wait=True)
        steps += 1

    print(t.bold_red("Fell into a hole.")) if reward == 0.0 else print(t.bold_green("Success!"))
    # time.sleep(.7)

    # how many times it fell into a hole vs success per episode
    agent.episode_plot.append(episode)
    agent.total_reward_plot.append(total_reward)
    agent.steps_plot.append(steps)

plot_seaborn(agent.episode_plot, agent.total_reward_plot, agent.steps_plot)
