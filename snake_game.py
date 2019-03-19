import random, time, os, copy, pprint, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from operator import add
from random import randint


BOARD_SIZE = 5
INIT_SNAKE_POSITIONS = [[math.floor(BOARD_SIZE/2), math.floor(BOARD_SIZE/2)]]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]
INIT_DIRECTION = 'up'
PREV_DIRECTION = 'up'
NUM_OF_ITERATIONS = 100
NUM_OF_INPUTS = 9
MODEL_NAME = 'my_weights.hdf5'
OPPOSITE_DIRECTION = {
    'up': 'down',
    'down': 'up',
    'right': 'left',
    'left': 'right'
}
RELATIVE_DIRECTION = {
    'forward': [1, 0, 0],
    'right': [0, 1, 0],
    'left': [0, 0, 1]
}

EPSILON_FACTOR = .998
MAX_EPSILON = .5
MIN_EPSILON = .1

GAMMA = .8


class Board:
    """Holds the board-list with the peices if any otherwise None"""

    board = None

    def __init__(self):
        self.board = [[None for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]

    def clear_board(self):
        for cell in self.board:
            cell = None

    def set_snake(self, game, tail=None):
        for body in game.snake.snake_body:
            # add body segments to board
            body_x = body[0]
            body_y = body[1]
            self.board[body_x][body_y] = body

        # remove tail segment from board
        if tail is not None:
            tail_x = tail[0]
            tail_y = tail[1]
            self.board[tail_x][tail_y] = None

    def set_apple(self, game):
        self.board[game.apple.apple[0]][game.apple.apple[1]] = game.apple

    def draw_board(self, board, game):
        for x, row in enumerate(board):
            for y, col in enumerate(row):
                if board[x][y] == game.snake.snake_body[0]:
                    print("[H]", end='')
                elif isinstance(board[x][y], Apple):
                    print("{}".format("[ ]" if col is None else "[A]"), end='')
                else:
                    print("{}".format("[ ]" if col is None else "[B]"), end='')
            print()

    def __str__(self):
        board_str = ""
        for row in self.board:
            for col in row:
                if col is None:
                    board_str += "[" + str(col) + "],"
                else:
                    board_str += str(col) + ","
            board_str += "\n"
        return board_str


class Agent:

    def __init__(self):
        self.current_reward = 0
        self.gamma = GAMMA
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.neural_network()
        self.model = self.neural_network("my_weights.hdf5")
        self.epsilon = MAX_EPSILON
        self.actual = []
        self.memory = []
        self.random_move = True
        self.random_moves = 1 # first move is 'considered' a random move (although it is always forced: 'up')

    def neural_network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=NUM_OF_INPUTS))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)

        return model

    def get_action(self, prev_state, prev_direction, possible_actions, game_counter):
        """returns a random action (up, down, right, left)"""
        if random.uniform(0, 1) < self.epsilon:
            current_direction = possible_actions[randint(0, 2)]
            # current_direction = to_categorical(randint(0, 2), num_classes=3) # I don't want to use this as I don't need to include the action
            self.random_move = True
            self.random_moves += 1
        else:
            prediction = self.model.predict(prev_state.reshape((1, NUM_OF_INPUTS)))
            # current_direction = to_categorical(np.argmax(prediction[0]), num_classes=3) # I don't want to use this as I don't need to include the action
            current_direction = possible_actions[np.argmax(prediction[0])] # will this work?
            self.random_move = False
        return current_direction

    def train_short_memory(self, prev_state, action, reward, current_state, game_over):
        target = reward
        if not game_over:
            target = reward + GAMMA * np.amax(self.model.predict(current_state.reshape((1, NUM_OF_INPUTS)))[0])
        target_f = self.model.predict(prev_state.reshape((1, NUM_OF_INPUTS)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(prev_state.reshape((1, NUM_OF_INPUTS)), target_f, epochs=1, verbose=0)

    def replay_new(self, memory):

        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory

        for state, action, reward, current_state, game_over in minibatch:
            target = reward
            if not game_over:
                target = reward + GAMMA * np.amax(self.model.predict(np.array([current_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)


class Snake:
    """Holds a list of all snake positions"""

    MAP = {
        'up': [-1, 0],
        'down': [1, 0],
        'right': [0, 1],
        'left': [0, -1],
    }

    def __init__(self, INIT_SNAKE_POSITIONS):
        self.snake_body = INIT_SNAKE_POSITIONS
        self.did_eat = False
        self.prev_direction = 'up'
        self.possible_actions = ['up', 'right', 'left']
        self.current_direction = INIT_DIRECTION # up
        self.current_relative_direction = 'forward'
        self.prev_relative_direction = 'forward'

    def set_current_relative_direction(self, current_direction, previous_direction):
        if current_direction == previous_direction:
            self.current_relative_direction = 'forward'
        elif current_direction == 'up':
            if previous_direction == 'right':
                self.current_relative_direction = 'left'
            if previous_direction == 'left':
                self.current_relative_direction = 'right'
        elif current_direction == 'down':
            if previous_direction == 'right':
                self.current_relative_direction = 'right'
            if previous_direction == 'left':
                self.current_relative_direction = 'left'
        elif current_direction == 'right':
            if previous_direction == 'up':
                self.current_relative_direction = 'right'
            if previous_direction == 'down':
                self.current_relative_direction = 'left'
        elif current_direction == 'left':
            if previous_direction == 'up':
                self.current_relative_direction = 'left'
            if previous_direction == 'down':
                self.current_relative_direction = 'right'

    def calc_destination(self, direction):
        """add diff_xy to snake_head"""
        diff_xy = self.MAP[direction]
        # print("diff_xy: {}".format(diff_xy))
        snake_head = self.snake_body[0]
        # print("snake_head: {}".format(snake_head))
        destination = [a_i + b_i for a_i, b_i in zip(diff_xy, snake_head)]
        # print("destination: {}".format(destination))
        return destination

    def valid_move(self, destination):
        if 0 <= destination[0] <= BOARD_SIZE-1 and 0 <= destination[1] <= BOARD_SIZE-1:
            return True
        else:
            # print("valid_move: Collision!")
            return False

    def self_collision(self, destination):
        if destination in self.snake_body:
            # if destination == self.snake_body[1]:
            #     print("NN MUST FIND ANOTHER ACTION/MOVE!!! MOVE NOT ALLOWED")
            return True
        else:
            return False

    def move(self, direction, game):
        destination = self.calc_destination(direction)
        tail = self.snake_body[-1]
        if self.apple_collision(destination, tail, game):
            tail = None
            game.apple = Apple()
            game.board.set_apple(game)
            self.did_eat = True
            game.score += 1
        del self.snake_body[-1]
        if not self.valid_move(destination):
            game.game_over = True
            tail = None
            # print("game_over={}".format(game.game_over))
            # print("APP: OutOfBoundsError: destination={} is outside the board.".format(destination))
        if self.self_collision(destination):
            game.game_over = True
            # print("game_over={}".format(game.game_over))
            # print("APP: SelfCollision: destination={} is part of the snake_body.".format(destination))
        else:
            self.snake_body.insert(0, destination)
        return tail

    def apple_collision(self, destination, tail, game):
        if destination == game.apple.apple:
            # print("apple_collision=True")
            self.snake_body.append(tail)  # problem: generalise tail input?
            return True
        else:
            return False

    def set_reward(self, game_over):
        self.reward = 0
        if game_over:
            self.reward = -10
        if self.did_eat:
            self.reward = 10
        return self.reward

    def get_close_sight(self, direction):
        neighbours = []
        if direction == 'up':
            neighbours.append(self.calc_destination('up')) # forward
            neighbours.append(self.calc_destination('right')) # right
            neighbours.append(self.calc_destination('left')) # left
        elif direction == 'down':
            neighbours.append(self.calc_destination('down')) # forward
            neighbours.append(self.calc_destination('left')) # right!
            neighbours.append(self.calc_destination('right')) # left!
        elif direction == 'right':
            neighbours.append(self.calc_destination('right')) # forward
            neighbours.append(self.calc_destination('down')) # right
            neighbours.append(self.calc_destination('up')) # left
        elif direction == 'left':
            neighbours.append(self.calc_destination('left')) # forward
            neighbours.append(self.calc_destination('up')) # right
            neighbours.append(self.calc_destination('down')) # left
        return neighbours

    def set_possible_actions(self, prev_direction):
        if prev_direction == 'up':
            self.possible_actions = ['up', 'right', 'left']
        elif prev_direction == 'down':
            self.possible_actions = ['down', 'left', 'right']
        elif prev_direction == 'right':
            self.possible_actions = ['right', 'down', 'up']
        elif prev_direction == 'left':
            self.possible_actions = ['left', 'up', 'down']

    def __str__(self):
        return "Snake({})".format(self.snake_body)

    def __repr__(self):
        return "Snke"


class Apple:
    """Initiates an apple with a random position, unless spesified"""

    def __init__(self, apple=None):
        if apple is None:
            self.apple = [
                random.randint(0, BOARD_SIZE-1),
                random.randint(0, BOARD_SIZE-1)]
        else:
            self.apple = apple

    def __repr__(self):
        return "Appl"


class Game:
    """Holds all game states from current game. Holds Board, Snake and Apple"""

    apple_count = 0
    game_over = False
    apples_eaten = 0
    tail = None

    def __init__(self):
        self.snake = Snake(INIT_SNAKE_POSITIONS[:])
        self.board = Board()
        self.apple = Apple()
        self.score = 0
        self.game_steps = 0
        self.prev_direction = ''
        self.game_states = []

    def rel_dir(self, direction):
        """absolute to relative direction mapping"""
        forward = direction
        if forward == 'up':
            right = 'right'
            left = 'left'
        elif forward == 'down':
            right = 'left'
            left = 'right'
        elif forward == 'right':
            right = 'down'
            left = 'up'
        elif forward == 'left':
            right = 'up'
            left = 'down'
        return forward, right, left

    def any_food(self, direction):
        """return True if there is any food in the given direction"""
        snake_head = self.snake.snake_body[0]
        check_for_food = []
        if not self.game_over:
            if direction == 'up':
                for i in range(1, (BOARD_SIZE - (BOARD_SIZE - snake_head[0]) + 1)):
                    check_for_food.append(isinstance(self.board.board[snake_head[0]-i][snake_head[1]], Apple))
            elif direction == 'down':
                for i in range(1, (BOARD_SIZE - (snake_head[0] + 1) + 1)):
                    check_for_food.append(isinstance(self.board.board[snake_head[0]+i][snake_head[1]], Apple))
            elif direction == 'right':
                for i in range(1, (BOARD_SIZE - (snake_head[1] + 1) + 1)):
                    check_for_food.append(isinstance(self.board.board[snake_head[0]][snake_head[1]+i], Apple))
            elif direction == 'left':
                for i in range(1, (BOARD_SIZE - (BOARD_SIZE - snake_head[1]) + 1)):
                    check_for_food.append(isinstance(self.board.board[snake_head[0]][snake_head[1]-i], Apple))
        return any(check_for_food)

    def get_state(self, direction, relative_direction):
        """Returns the state of the game in an ordered and onehot encoded list of booleans (1 or 0). This will be the input of the neural network"""
        state = []

        # dangers
        neighbours = self.snake.get_close_sight(direction)
        for n in neighbours:
            if self.snake.valid_move(n) and not self.snake.self_collision(n):
                state.append(False) # no danger
            else:
                state.append(True) # danger

        # food
        forward, right, left = self.rel_dir(direction)
        state.append(self.any_food(forward))
        state.append(self.any_food(right))
        state.append(self.any_food(left))

        # move
        state.extend(RELATIVE_DIRECTION[relative_direction])

        for i, s in enumerate(state):
            if s:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def update(self, direction):  # Update Game State
        """Takes a direction and executes move"""
        self.board.set_apple(self)
        self.tail = self.snake.move(direction, self)
        self.board.clear_board()
        try:
            self.board.set_snake(self, self.tail)
        except Exception as e:
            pass
        self.game_states.append(copy.deepcopy(self.board.board))
        self.board.draw_board(self.board.board, self)

    def initial_update(self, simulation, agent, game):
        prev_state = self.get_state(PREV_DIRECTION, game.snake.prev_relative_direction)
        self.snake.set_possible_actions(self.prev_direction)
        self.snake.current_direction = INIT_DIRECTION
        game.snake.set_current_relative_direction(self.snake.current_direction, PREV_DIRECTION)
        self.update(INIT_DIRECTION)
        current_state = self.get_state(self.snake.current_direction, self.snake.current_relative_direction)
        reward = self.snake.set_reward(self.game_over)
        agent.train_short_memory(prev_state, RELATIVE_DIRECTION[game.snake.current_relative_direction], reward, current_state, self.game_over)
        agent.memory.append((prev_state, RELATIVE_DIRECTION[game.snake.current_relative_direction], reward, current_state, self.game_over))
        simulation.set_high_score(self.score)
        self.game_steps += 1
        self.prev_direction = INIT_DIRECTION
        print('current_state:\t{}'.format(current_state))
        print('{} ({})\trandom_move={}\tepsilon={}%'.format(game.snake.current_direction, game.snake.current_relative_direction, agent.random_move, int(agent.epsilon*100)))
        print('memory_length={}'.format(len(agent.memory)))
        # pprint.pprint(agent.memory)
        time.sleep(0.05)


class Simulation:

    def __init__(self):
        self.name = 'my_simulation'
        self.games = []
        self.game_counter = 0
        self.score_plot = []
        self.counter_plot = []
        self.agent = Agent()
        self.high_score = 0
        self.total_game_steps = 0
        self.total_epsilon = 0
        self.avg_steps = 1
        self.avg_epsilon = 0

    def plot_seaborn(self, array_counter, array_score):
        sns.set(color_codes=True)
        ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
        ax.set(xlabel='games', ylabel='score')
        plt.savefig('plot_figure.png')
        plt.show()

    def get_best_game(self):
        if self.games:
            best_game = self.games[0]
            for game in self.games:
                if game.score >= best_game.score:
                    best_game = game
        else:
            print("Simulation don't have any games. Run it with simulation.run() to gather some games.")
        return best_game

    def replay_game(self, game):
        for current_board in game.game_states:
            game.board.draw_board(current_board, game)
            print()
            time.sleep(0.05)

    def set_high_score(self, score):
        if score >= self.high_score:
            self.high_score = score

    def run(self, num_of_iterations):

        while self.game_counter < num_of_iterations: # simulation-loop (continue training until user stops simulation)
            print('====== game {} ======'.format(self.game_counter))
            game = Game()
            game.board.set_snake(game)
            game.initial_update(self, self.agent, game)

            while not game.game_over: # game-loop
                prev_state = game.get_state(game.prev_direction, game.snake.prev_relative_direction)
                game.snake.set_possible_actions(game.prev_direction)
                game.snake.current_direction = self.agent.get_action(prev_state, game.prev_direction, game.snake.possible_actions, self.game_counter)
                game.snake.set_current_relative_direction(game.snake.current_direction, game.prev_direction)
                game.update(game.snake.current_direction)
                current_state = game.get_state(game.snake.current_direction, game.snake.current_relative_direction)
                reward = game.snake.set_reward(game.game_over)
                self.agent.train_short_memory(prev_state, RELATIVE_DIRECTION[game.snake.current_relative_direction], reward, current_state, game.game_over)
                self.agent.memory.append((prev_state, RELATIVE_DIRECTION[game.snake.current_relative_direction], reward, current_state, game.game_over))
                self.set_high_score(game.score)
                game.prev_direction = game.snake.current_direction
                game.snake.prev_relative_direction = game.snake.current_relative_direction
                print('current_state:\t{}'.format(current_state))
                print('{} ({})\trandom_move={}\tepsilon={}%'.format(game.snake.current_direction, game.snake.current_relative_direction, self.agent.random_move, int(self.agent.epsilon*100)))
                print('memory_length={}'.format(len(self.agent.memory)))
                # pprint.pprint(self.agent.memory)
                time.sleep(0.05)
                game.game_steps += 1
                if game.game_steps > 500:
                    game.game_over = True
                if self.agent.epsilon > MIN_EPSILON:
                    self.agent.epsilon *= EPSILON_FACTOR

            self.total_game_steps += game.game_steps
            self.total_epsilon += self.agent.epsilon
            self.agent.replay_new(self.agent.memory)
            self.games.append(game) # append game to the list of games
            self.score_plot.append(game.score)
            self.counter_plot.append(self.game_counter)
            self.avg_steps = self.total_game_steps / (self.game_counter + 1)
            self.avg_epsilon = self.total_epsilon / (self.game_counter + 1)
            print()
            print('avg: avg_steps:{}\tavg_epsilon:{}%'.format(round(self.avg_steps, 2), round(self.avg_epsilon, 2)))
            print('current: game:{}\tscore:{}\thigh_score:{}\tpredicted_moves:{}/{} ({})%'.format(self.game_counter, game.score, self.high_score, (game.game_steps - self.agent.random_moves), game.game_steps, int(((game.game_steps - self.agent.random_moves) / game.game_steps)*100)))
            print('====== end of game ======')
            print()
            self.agent.random_moves = 0
            time.sleep(1.2)
            self.game_counter += 1

        self.agent.model.save_weights('my_weights.hdf5') # save model
        self.plot_seaborn(self.counter_plot, self.score_plot)
        print('Simulation done: Saving Model as:', MODEL_NAME)


import ipdb; ipdb.set_trace()
simulation = Simulation()
simulation.run(NUM_OF_ITERATIONS)
best_game = simulation.get_best_game()
simulation.replay_game(best_game)
