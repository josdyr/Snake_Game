import random
import os
import time

BOARD_SIZE = 10
INIT_SNAKE_POSITIONS = [[0,0], [1,0], [2,0]]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]


# class Cell:
#     piece = None

class Board:
    board = [[None for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]

    def set_snake(self):
        for body in snake.snake_body:
            x = body[0]
            y = body[1]

            self.board[x][y] = body

    def draw_board(self):
        for x, row in enumerate(self.board):
            for y, col in enumerate(row):
                if self.board[x][y] == snake.snake_head:
                    print("[H]", end='')
                else:
                    print("{}".format("[ ]" if col is None else "[B]"), end='')
            print()

class Game:
    board = Board()

    def Start(self):
        while True:
            for body in bodies_to_be_added:
                snake.add_body(body)

                os.system('clear')
                game.board.set_snake()
                game.board.draw_board()

                time.sleep(1)

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

class Apple:
    apple = [random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1)]


# import pdb; pdb.set_trace()
bodies_to_be_added = [[3,0], [4,0], [5,0]]

snake = Snake(INIT_SNAKE_POSITIONS)
game = Game()
game.Start()

# game.board.set_snake()
# game.board.draw_board()
