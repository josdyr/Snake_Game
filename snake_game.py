import random


BOARD_SIZE = 4
INIT_SNAKE_POSITIONS = [[0, 0], [1, 0], [2, 0]]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]
GENERATIONS = 10


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
    score = 0
    game_over = False
    apples_eaten = 0
    game_states = []
    tail = None

    def __init__(self):
        self.board = Board()
        self.snake = Snake(INIT_SNAKE_POSITIONS)
        self.apple = Apple()

    def update(self, direction):  # Update Game State
        """Takes a direction and executes move"""
        self.board.set_apple()
        self.tail = self.snake.move(direction)
        if self.game_over:
            exit()
        self.board.reset_board(self.tail)
        self.game_states.append(self.board)
        self.board.draw_board()


game = Game()
