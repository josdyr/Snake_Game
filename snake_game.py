import random
import os
import time


BOARD_SIZE = 10


class Board:
    board = [[None for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]

    def set_snake(self):
        for body in snake.snake_body:

            # add body segments to board
            if isinstance(body, Body):
                self.board[body.x][body.y] = body
            else:
                self.board[body.x][body.y] = None

    def set_apple(self):
        x = apple.apple[0]
        y = apple.apple[1]
        self.board[x][y] = apple

    def draw_board(self):
        # os.system('clear')
        for x, row in enumerate(self.board):
            for y, col in enumerate(row):
                if isinstance(col, Body):
                    print("[{}]".format(col.char), end='')
                elif isinstance(col, Apple):
                    print("{}".format("[ ]" if col is None else "[A]"), end='')
                else:
                    print("{}".format("[ ]" if col is None else "[{}]".format(col.char)), end='')
            print()

class Game:
    board = Board()

    def start(self):

        i = 0
        while True:
            for body in bodies_to_be_added:
                print("Snake:{}".format([str(body) for body in snake.snake_body]))
                snake.move(MOVES[i])
                # snake.add_body(body)
                game.board.set_snake()
                game.board.draw_board()
                print("Snake:{}".format([str(body) for body in snake.snake_body]))
                time.sleep(1)
            game.board.set_apple()
            time.sleep(1)
            i += 1
            break

class Snake:
    def __init__(self, INIT_SNAKE_POSITIONS):
        self.snake_body = INIT_SNAKE_POSITIONS
        self.snake_head = self.snake_body[0]

    def reset_snake(self):
        self.snake_body = INIT_SNAKE_POSITIONS

    def add_body(self, body):
        self.snake_body.append(body)

    def move(self, destination):
        del snake.snake_body[-1]
        snake.snake_body[-1].char = 'B'
        new_head = Body(destination[0], destination[1])
        new_head.char = 'H'
        snake.snake_body.insert(0, new_head)

class Body:
    char = 'B'

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "[{},{}]".format(self.x, self.y)

class Apple:
    char = 'A'

    def __init__(self):
        self.apple = [random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1)]


# PARAMETERS
INIT_SNAKE_POSITIONS = [Body(0,0), Body(1,0), Body(2,0)]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]
MOVES = [[0,1], [0,2]]


# START
bodies_to_be_added = [[3,0], [4,0]]

snake = Snake(INIT_SNAKE_POSITIONS)
# snake.snake_body[0].char = 'H'

game = Game()
apple = Apple()

# os.system('clear')
game.board.set_snake()
game.board.set_apple()
game.board.draw_board()

time.sleep(1)

import pdb; pdb.set_trace()
game.start()
