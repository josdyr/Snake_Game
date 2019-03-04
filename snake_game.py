import random
from random import randint
import numpy as np
import pandas as pd
from operator import add
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical


BOARD_SIZE = 4
INIT_SNAKE_POSITIONS = [[0, 0], [1, 0], [2, 0]]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]
INIT_DIRECTION = 'right'
PREV_DIRECTION = 'up'

GENERATIONS = 10
NUM_OF_INPUTS = 6
MODEL_NAME = 'my_weights.hdf5'
ACTIONS = ['up', 'down', 'right', 'left']


class Board:
    """Holds the board-list with the peices if any otherwise None"""

    board = [[None for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]

    def clear_board(self):
        for cell in self.board:
            cell = None

    def set_snake(self, tail=None):
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

    def set_apple(self):
        self.board[game.apple.apple[0]][game.apple.apple[1]] = game.apple

    def draw_board(self):
        for x, row in enumerate(self.board):
            for y, col in enumerate(row):
                if self.board[x][y] == game.snake.snake_body[0]:
                    print("[H]", end='')
                elif isinstance(self.board[x][y], Apple):
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


class SnakeAgent:
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

        self.current_reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame() # ???
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.neural_network()
        # self.model = self.neural_network("my_weights.hdf5")
        self.epsilon = 80
        self.actual = []
        self.memory = []

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

    def move(self, direction):
        destination = self.calc_destination(direction)
        tail = self.snake_body[-1]
        if self.apple_collision(destination, tail):
            tail = None
            game.apple = Apple()
            game.board.set_apple()
            self.did_eat = True
        del self.snake_body[-1]
        if not self.valid_move(destination):
            game.game_over = True
            tail = None
            print("game_over={}".format(game.game_over))
            print("APP: OutOfBoundsError: destination={} is outside the board.".format(destination))
        if self.self_collision(destination):
            game.game_over = True
            print("game_over={}".format(game.game_over))
            print("APP: SelfCollision: destination={} is part of the snake_body.".format(destination))
        else:
            self.snake_body.insert(0, destination)
        return tail

    def apple_collision(self, destination, tail):
        if destination == game.apple.apple:
            print("apple_collision=True")
            self.snake_body.append(tail)  # problem: generalise tail input?
            return True
        else:
            return False

    def get_action(self, prev_state):
        """returns a random action (forward, right, left)"""
        if random.randint(0, 100) < self.epsilon:
            current_direction = ACTIONS[randint(0, 3)]
            # current_direction = to_categorical(randint(0, 2), num_classes=3)
        else:
            prediction = self.model.predict(prev_state.reshape((1, NUM_OF_INPUTS)))
            # current_direction = to_categorical(np.argmax(prediction[0]), num_classes=3)
            current_direction = ACTIONS[np.argmax(prediction[0])] # find out what this does
        self.epsilon -= 1

        # check if snake is moving back onto itself (if so, then pick another action)
        destination = self.calc_destination(current_direction)
        if destination == self.snake_body[1]:
            self.get_action(prev_state)
        else:
            return current_direction

    def set_reward(self, game_over):
        self.reward = 0
        if game_over:
            self.reward = -10
        if self.did_eat:
            self.reward = 10
        return self.reward

    def train_short_memory(self, prev_state, action, reward, current_state, game_over):
        target = reward
        if not game_over:
            target = reward + self.gamma * np.amax(self.model.predict(current_state.reshape((1, NUM_OF_INPUTS)))[0])
        target_f = self.model.predict(prev_state.reshape((1, NUM_OF_INPUTS)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(prev_state.reshape((1, NUM_OF_INPUTS)), target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, current_state, done):
        self.memory.append((state, action, reward, current_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory

        for state, action, reward, current_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([current_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

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

    def __repr__(self):
        return "Snake({})".format(self.snake_body)


class Apple:
    """Initiates an apple with a random position, unless spesified"""

    def __init__(self, apple=None):
        if apple is None:
            self.apple = [
                random.randint(0, BOARD_SIZE-1),
                random.randint(0, BOARD_SIZE-1)]
        else:
            self.apple = apple

    def __str__(self):
        return "[Appl]"


class Game:
    """Holds all game states from current game. Holds Board, Snake and Apple"""

    apple_count = 0
    game_over = False
    apples_eaten = 0
    game_states = []
    tail = None

    def __init__(self):
        self.snake = SnakeAgent(INIT_SNAKE_POSITIONS)
        self.board = Board()
        self.apple = Apple()
        self.score = 0
        self.record = 0
        self.game_steps = 0
        self.prev_direction = ''

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

    def get_state(self, direction):
        """returns the state of the game. Will return a list of onehot encoded booleans (1 or 0). The list will then be the input of the neural network"""
        # state: order matter! -> forward, right, left (relative)
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

        for i, s in enumerate(state):
            if s:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def update(self, direction):  # Update Game State
        """Takes a direction and executes move"""
        self.board.set_apple()
        self.tail = self.snake.move(direction)
        self.board.clear_board()
        self.board.set_snake(self.tail)
        self.game_states.append(self.board)
        self.board.draw_board()

    def initial_update(self):
        prev_state = self.get_state(PREV_DIRECTION)
        self.snake.current_direction = INIT_DIRECTION
        self.update(INIT_DIRECTION)
        current_state = self.get_state(self.snake.current_direction)
        reward = self.snake.set_reward(self.game_over)
        self.snake.train_short_memory(prev_state, self.snake.current_direction, reward, current_state, self.game_over)
        self.snake.remember(prev_state, self.snake.current_direction, reward, current_state, self.game_over)
        self.record = get_record(self.score, self.record)
        self.game_steps += 1
        game.prev_direction = PREV_DIRECTION


def get_record(score, record):
        if score >= record:
            return score
        else:
            return record


game_counter = 0
while True: # simulation-loop (continue training until user stops simulation)
    import ipdb; ipdb.set_trace()
    games = []
    game = Game()
    game.board.set_snake()
    game.board.draw_board()
    game.initial_update()
    while not game.game_over: # game-loop
        prev_state = game.get_state(game.prev_direction)
        game.snake.current_direction = game.snake.get_action(prev_state)
        print(game.snake.current_direction)
        import ipdb; ipdb.set_trace()
        game.update(game.snake.current_direction)
        current_state = game.get_state(game.snake.current_direction)
        reward = game.snake.set_reward(game.game_over)
        game.snake.train_short_memory(prev_state, game.snake.current_direction, reward, current_state, game.game_over)
        game.snake.remember(prev_state, game.snake.current_direction, reward, current_state, game.game_over)
        game.record = get_record(game.score, game.record)
        game.game_steps += 1
        game.prev_direction = game.snake.current_direction

    game.snake.replay_new(game.snake.memory)
    print('Game', game_counter, ', Score', game.score)
    games.append(game) # append game to the list of games
    print('Current game done: Saving game and launching a new one.')
    game_counter += 1

game.snake.model.save_weights('my_weights.hdf5') # # save model
print('Simulation done: Saving Model as:', MODEL_NAME)
