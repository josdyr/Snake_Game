import random
import os
import time


GENERATIONS = 1000
BOARD_SIZE = 10
INIT_SNAKE_POSITIONS = [[0,0], [1,0], [2,0]]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]
MOVES = ['right','right','right','down','down','left','down','down', 'down','left','left','left','left']


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

    def draw_board(self):
        # os.system('clear')
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

class Snake:
    """Holds a list of all snake positions"""

    MAP = {
        'up': [-1,0],
        'down': [1,0],
        'right': [0,1],
        'left': [0,-1],
    }

    def __init__(self, INIT_SNAKE_POSITIONS):
        self.snake_body = INIT_SNAKE_POSITIONS

    def reset_snake(self):
        self.snake_body = INIT_SNAKE_POSITIONS

    def add_body(self, body):
        self.snake_body.append(body)

    def calc_destination(self, direction):
        """add diff_xy to snake_head"""
        diff_xy = self.MAP[direction]; print("diff_xy: {}".format(diff_xy))
        snake_head = self.snake_body[0]; print("snake_head: {}".format(snake_head))
        destination = [a_i + b_i for a_i, b_i in zip(diff_xy, snake_head)]
        print("destination: {}".format(destination))
        return destination

    def valid_move(self, destination):
        if 0 <= destination[0] and destination[1] <= 9:
            return True
        else:
            return False

    def move(self, direction):
        destination = self.calc_destination(direction)
        tail = self.snake_body[-1]
        if self.apple_collision(destination, tail):
            tail = None
            self.apple = Apple(unrandom_apples_list[game.apples_eaten])
            self.apple.set_apple()
            game.apples_eaten += 1
        del self.snake_body[-1]
        if not self.valid_move(destination):
            game.game_over = True
            tail = None
            print("game_over={}".format(game.game_over))
            print("APP: OutOfBoundsError: destination={} is outside the board.".format(destination))
        else:
            self.snake_body.insert(0, destination)
        return tail

    def wall_collision(self):
        pass

    def apple_collision(self, destination, tail):
        if destination == game.apple.apple:
            print("apple_collision=True")
            self.snake_body.append(tail) # problem: generalise tail input?
            return True
        else:
            return False

    def self_collision(self):
        pass

    def __repr__(self):
        return "Snake({})".format(self.snake_body)

class Apple:
    """Initiates an apple with a random position, unless spesified"""

    apple = [random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1)]

    def __init__(self, apple=apple):
        self.apple = [apple[0], apple[1]]

    def set_apple(self):
        x = self.apple[0]
        y = self.apple[1]
        game.board.board[x][y] = game.apple

    def __str__(self):
        return "[Appl]"

class Game:
    """Holds all game states from current game"""

    board = Board()
    apple = None

    apple_count = 0
    score = 0
    game_over = False
    loop_count = 0
    apples_eaten = 0
    current_move = None
    game_states = []

    def __init__(self):
        self.snake = Snake(INIT_SNAKE_POSITIONS)

    def run_game(self):

        self.apple = Apple([0,3])
        self.apple.set_apple()

        while not self.game_over and self.loop_count <= len(MOVES)-1:
            self.current_move = MOVES[self.loop_count]
            tail = self.snake.move(self.current_move)
            self.board.reset_board(tail)
            self.game_states.append(self.board)
            self.board.draw_board()
            print(repr(self.snake))
            print(str(self.board))
            time.sleep(1)
            self.loop_count += 1


if __name__ == "__main__":

    perception_forward = ['apple']
    perception_left = ['wall']
    perception_right = ['self']

    unrandom_apples_list = [[2,2],[4,2],[4,2]]
    all_games = []
    game_count = 0

    while game_count <= GENERATIONS:

        game = Game()
        game.run_game()
        all_games.append(game)
        game_count += 1
