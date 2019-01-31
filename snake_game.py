import random
import os
import time

BOARD_SIZE = 10
INIT_SNAKE_POSITIONS = [[0,0], [1,0], [2,0]]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]


class Board:
    """Holds the board-list with the peices if any otherwise None"""

    board = [[None for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]

    def clear_board(self):
        for cell in self.board:
            cell = None

    def reset_board(self, tail=None):
        self.clear_board()
        for body in snake.snake_body:
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
        x = apple.apple[0]
        y = apple.apple[1]
        self.board[x][y] = apple

    def draw_board(self):
        # os.system('clear')
        for x, row in enumerate(self.board):
            for y, col in enumerate(row):
                if self.board[x][y] == snake.snake_body[0]:
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
        tail = snake.snake_body[-1]
        del snake.snake_body[-1]
        if not self.valid_move(destination):
            game.game_over = True
            tail = None
            print("game_over={}".format(game.game_over))
            print("APP: OutOfBoundsError: destination={} is outside the board.".format(destination))
        else:
            snake.snake_body.insert(0, destination)
        return tail

    def wall_collision(self):
        pass

    def apple_collision(self):
        if snake.snake_body[0] == apple.apple:
            print("apple_collision=True")
            snake.snake_body.append([0,0]) # problem: generalise tail input?
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

    def __str__(self):
        return "[Appl]"

class Game:
    """Holds the board"""

    board = Board()

    game_over = False
    loop_count = 0
    current_move = None

    def start(self):
        while not self.game_over and self.loop_count <= len(MOVES)-1:
            self.current_move = MOVES[self.loop_count]
            # import pdb; pdb.set_trace()
            tail = snake.move(self.current_move)
            if snake.apple_collision():
                tail = None

            game.board.reset_board(tail)
            game_states.append(game.board)
            game.board.draw_board()

            print(repr(snake))
            print(str(game.board))

            time.sleep(1)
            self.loop_count += 1

MOVES = ['right','right','right','down']
game_states = []
snake = Snake(INIT_SNAKE_POSITIONS)
game = Game()
apple = Apple([0,3]); game.board.set_apple()

# os.system('clear')
game.board.reset_board()
game.board.draw_board()

print(repr(snake))
print(game.board)

time.sleep(1)
game.start()
