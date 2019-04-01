import unittest, math
from snake_game import Snake, Game


BOARD_SIZE = 5
INIT_SNAKE_POSITIONS = [[math.floor(BOARD_SIZE/2), math.floor(BOARD_SIZE/2)]]
INIT_POSITION = INIT_SNAKE_POSITIONS[0]

# import ipdb; ipdb.set_trace()
# snake = Snake(INIT_SNAKE_POSITIONS[:])
game = Game()

class TestMethods(unittest.TestCase):

    def test_calc_destination_up(self):
        game.snake.snake_body = [[2, 2]]
        snake_head = game.snake.snake_body[0]
        destination = game.snake.calc_destination('up')
        self.assertEqual(destination, [1, 2])

    def test_calc_destination_down(self):
        game.snake.snake_body = [[2, 2]]
        snake_head = game.snake.snake_body[0]
        destination = game.snake.calc_destination('down')
        self.assertEqual(destination, [3, 2])

    def test_calc_destination_right(self):
        game.snake.snake_body = [[2, 2]]
        snake_head = game.snake.snake_body[0]
        destination = game.snake.calc_destination('right')
        self.assertEqual(destination, [2, 3])

    def test_calc_destination_left(self):
        game.snake.snake_body = [[2, 2]]
        snake_head = game.snake.snake_body[0]
        destination = game.snake.calc_destination('left')
        self.assertEqual(destination, [2, 1])

    def test_get_close_sight_up(self):
        neighbours = game.snake.get_close_sight('up')
        self.assertEqual(neighbours, [game.snake.calc_destination('up'), game.snake.calc_destination('right'), game.snake.calc_destination('left')])

    def test_get_close_sight_down(self):
        neighbours = game.snake.get_close_sight('down')
        self.assertEqual(neighbours, [game.snake.calc_destination('down'), game.snake.calc_destination('left'), game.snake.calc_destination('right')])

    def test_get_close_sight_right(self):
        neighbours = game.snake.get_close_sight('right')
        self.assertEqual(neighbours, [game.snake.calc_destination('right'), game.snake.calc_destination('down'), game.snake.calc_destination('up')])

    def test_get_close_sight_left(self):
        neighbours = game.snake.get_close_sight('left')
        self.assertEqual(neighbours, [game.snake.calc_destination('left'), game.snake.calc_destination('up'), game.snake.calc_destination('down')])

if __name__ == '__main__':
    unittest.main()
