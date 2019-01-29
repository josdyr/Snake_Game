import random
import os
import time

BOARD_SIZE = 10
INIT_SNAKE_POSITIONS = [[0,0], [1,0], [2,0]]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]


class Board:
    board = [[None for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]

    def set_snake(self, tail=None):
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
                if self.board[x][y] == snake.snake_head:
                    print("[H]", end='')
                elif isinstance(self.board[x][y], Apple):
                    print("{}".format("[ ]" if col is None else "[A]"), end='')
                else:
                    print("{}".format("[ ]" if col is None else "[B]"), end='')
            print()

class Game:
    board = Board()

    def start(self):

        while True:
            for body in bodies_to_be_added:
                tail = snake.move([0,1])
                # snake.add_body(body)

                game.board.set_snake(tail=tail)
                game.board.draw_board()
                time.sleep(1)
            game.board.set_apple()
            time.sleep(1)
            break

class Snake:
    snake_head_char = "H"
    snake_body_char = "B"

    def __init__(self, INIT_SNAKE_POSITIONS):
        self.snake_body = INIT_SNAKE_POSITIONS
        self.snake_head = self.snake_body[0]

    def reset_snake(self):
        self.snake_body = INIT_SNAKE_POSITIONS

    def add_body(self, body):
        self.snake_body.append(body)

    def move(self, destination):
        tail = snake.snake_body[-1]
        del snake.snake_body[-1]
        snake.snake_body.insert(0, destination)
        return tail

    def set_snake_head():
        pass

class Apple:

    def __init__(self):
        self.apple = [random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1)]



bodies_to_be_added = [[3,0]]

snake = Snake(INIT_SNAKE_POSITIONS)
game = Game()
apple = Apple()

# os.system('clear')
# import pdb; pdb.set_trace()
game.board.set_snake()
game.board.set_apple()
game.board.draw_board()
time.sleep(1)
game.start()
