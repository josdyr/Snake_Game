import random


BOARD_SIZE = 10


class Cell:
    piece = None

class Board:
    board = [[Cell() for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]

    def draw_board(self):
        for row in self.board:
            for cell in row:
                print("hi")
            print()

class Game:
    board = Board()

    def Start(self):
        while True:
            #run Game
            pass

class Snake:
    snake_body = [None]
    snake_head = snake_body[0]

class Apple:
    apple = [random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1)]


game = Game()
game.board.draw_board()
