import random
import os
import time

BOARD_SIZE = 10
INIT_SNAKE_POSITIONS = [[0,0], [1,0], [2,0]]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]


class Board:
    """Holds the board-list with the peices if any otherwise None"""

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

    def __init__(self, INIT_SNAKE_POSITIONS):
        self.snake_body = INIT_SNAKE_POSITIONS

    def reset_snake(self):
        self.snake_body = INIT_SNAKE_POSITIONS

    def add_body(self, body):
        self.snake_body.append(body)

    def move(self, destination):
        tail = snake.snake_body[-1]
        del snake.snake_body[-1]
        try:
            snake.snake_body.insert(0, destination)
        except Exception as e:
            raise
        return tail

    def wall_collision(self):
        pass

    def apple_collision(self):
        pass

    def self_collision(self):
        pass

    def __repr__(self):
        return "Snake({})".format(self.snake_body)

class Apple:

    def __init__(self):
        self.apple = [random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1)]

    def __str__(self):
        return "[Appl]"

class Game:
    """Holds the board"""

    board = Board()

    def start(self):

        i = 0
        running = True
        while running:
            for body in bodies_to_be_added:

                tail = snake.move(MOVES[i])
                game.board.set_snake(tail=tail)
                game_states.append(game.board)

                game.board.draw_board()
                print(repr(snake))
                print(str(game.board))
                print(game.board)

                time.sleep(1)
                i += 1

            game.board.set_apple()
            time.sleep(1)
            break


bodies_to_be_added = [[3,0],[4,0],[5,0],[6,0]]
MOVES = [[0,1],[0,2],[1,2],[2,2]]
# MOVES = ['r','f','r','f']

# holds snapshots of all previous game states
game_states = []

snake = Snake(INIT_SNAKE_POSITIONS)
game = Game()
apple = Apple()

# os.system('clear')
game.board.set_snake()
game.board.set_apple()
game.board.draw_board()
print(repr(snake))
print(str(game.board))
print(game.board)

time.sleep(1)
game.start()
