import random
import numpy as np


BOARD_SIZE = 4
INIT_SNAKE_POSITIONS = [[0, 0], [1, 0], [2, 0]]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]
GENERATIONS = 10
NUM_OF_INPUTS = 9
MODEL_NAME = 'my_weights.hdf5'


class Board:
    """Holds the board-list with the peices if any otherwise None"""

    board = [[None for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]

    def clear_board(self):
        for cell in self.board:
            cell = None

    def reset_board(self, tail=None):
        self.clear_board()
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

        self.current_reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame() # ???
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        self.model = self.network("my_weights.hdf5")
        self.epsilon = 80
        self.actual = []
        self.memory = []

    def neural_network(self, weights=None):
        self.model = Sequential()
        self.model.add(Dense(output_dim=120, activation='relu', input_dim=NUM_OF_INPUTS))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(output_dim=120, activation='relu'))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(output_dim=120, activation='relu'))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(output_dim=3, activation='softmax'))
        self.opt = Adam(self.learning_rate)
        self.model.compile(loss='mse', optimizer=opt)

    def calc_destination(self, direction):
        """add diff_xy to snake_head"""
        diff_xy = self.MAP[direction]
        print("diff_xy: {}".format(diff_xy))
        snake_head = self.snake_body[0]
        print("snake_head: {}".format(snake_head))
        destination = [a_i + b_i for a_i, b_i in zip(diff_xy, snake_head)]
        print("destination: {}".format(destination))
        return destination

    def valid_move(self, destination):
        if 0 <= destination[0] <= 9 and 0 <= destination[1] <= 9:
            return True
        else:
            print("Collision!")
            return False

    def self_collision(self, destination):
        if destination in self.snake_body:
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
            print(
                "APP: OutOfBoundsError: destination={} is outside the board."
                .format(destination))
        if self.self_collision(destination):
            game.game_over = True
            print("game_over={}".format(game.game_over))
            print(
                "APP: SelfCollision: destination={} is part of the snake_body."
                .format(destination))
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

    def get_action(self):
        if random.randint(0, 100) < self.epsilon:
            current_action = to_categorical(randint(0, 2), num_classes=3)
        else:
            prediction = self.model.predict(prev_state.reshape((1, NUM_OF_INPUTS)))
            current_action = to_categorical(np.argmax(prediction[0]), num_classes=3)
        self.epsilon -= 1
        return current_action

    def set_reward(self, game_over):
        self.reward = 0
        if game_over:
            self.reward = -10
        if self.did_eat:
            self.reward = 10
        return self.reward

    def train_short_memory(self, prev_state, action, reward, current_new_state, game_over):
        target = reward
        if not game_over:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, NUM_OF_INPUTS)))[0])
        target_f = self.model.predict(state.reshape((1, NUM_OF_INPUTS)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, NUM_OF_INPUTS)), target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

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
        self.board = Board()
        self.snake = SnakeAgent(INIT_SNAKE_POSITIONS)
        self.apple = Apple()
        self.score = 0

    def initial_update(self):
        prev_state = None
        pass

    def get_state(self):
        """returns the state of the game. Will return a list of onehot encoded booleans (1/0). The list will then be the input of the neural network"""
        pass
        # state = [d_front, d_right, d_left, f_front, f_right, f_left, m_front, m_right, m_left]

        # return np.asarray(state)

    def update(self, direction):  # Update Game State
        """Takes a direction and executes move"""
        self.board.set_apple()
        self.tail = self.snake.move(direction)
        self.board.reset_board(self.tail)
        self.game_states.append(self.board)
        self.board.draw_board()


def get_record(score, record):
        if score >= record:
            return score
        else:
            return record


import ipdb; ipdb.set_trace()

while True: # simulation-loop (continue training until user stops simulation)

    # make a game for each simulation-iteration
    game = Game()

    # initial update (the very first update/game-step/action/game-iteration)
    game.initial_update()

    # the rest of the updates/game-steps/actions/game-iterations
    while not game.game_over:

        # get state (which will be the input to the neural network)
        prev_state = game.get_state()

        # preform move of get_action() (predict action and update game state)
        game.update(get_action())

        # get the new game-state (now that the move is acted upon the environment)
        current_new_state = game.get_state()

        # get reward from current_new_state (apple=10pts, wall/self=-10pts)
        reward = game.snake.set_reward(game.game_over)

        # train short memory based on the new action and new state
        game.snake.train_short_memory(prev_state, action, reward, current_new_state, game.game_over)

        # save new data in a long term memory
        game.snake.remember(prev_state, action, reward, current_new_state, game.game_over)

        # if there is a new record, then save it
        record = get_record()

    game.snake.replay_new()
    game_counter += 1
    print('Game', game_counter, ', Score', game.score)
    games.append(game) # append game to the list of games
    print('Current game done: Saving game and launching a new one.')

# save model
game.snake.model.save_weights('my_weights.hdf5') # swap with MODEL_NAME as arg
print('Simulation done: Saving Model as:', MODEL_NAME)
