import numpy as np
import random
from tetraminos import tetraminos
from enum import Enum

WELL_DEPTH = 8
WELL_WIDTH = 10
FRAMES_PER_DROP = 3
TETRAMINO_START_X_POS = 3


SHAPES_USED = 1


class Action(Enum):
    LEFT = 1
    RIGHT = 2
    ROTATE_LEFT = 3
    ROTATE_RIGHT = 4
    DO_NOTHING = 5


class Game:
    def __init__(self):
        self.terminated = False
        self.well = np.zeros((WELL_DEPTH, WELL_WIDTH))

        self.emerge()
        self.frame = 0

        self.score = 0

    def get_state(self):
        state = self.well.copy()

        (th, tw) = self.tetramino.shape
        for y in range(th):
            for x in range(tw):
                if (self.tetramino[y][x] == 1):
                    state[self.tetramino_y + y][self.tetramino_x + x] = 1
        return state.reshape(1, WELL_DEPTH * WELL_WIDTH)[0]

    def has_terminated(self):
        return self.terminated

    def next(self, action):
        self.apply_action(action)
        self.frame += 1
        self.score += 1
        if (self.frame == FRAMES_PER_DROP):
            if (self.has_room_below()):
                self.sink()
            else:
                self.land()
                self.emerge()
                if (self.is_game_over()):
                    self.terminated = True
                    return 1  # todo: ?
            self.frame = 0
        return 1  # todo: reward

    def get_random_shape(self):
        return random.randint(0, SHAPES_USED - 1)

    def apply_action(self, action):
        if action == Action.LEFT and self.has_room_left():
            self.tetramino_x -= 1
            return
        elif action == Action.RIGHT and self.has_room_right():
            self.tetramino_x += 1
            return
        elif action == Action.ROTATE_LEFT and self.would_fit(self.dec_angle(self.tetramino_angle)):
            self.tetramino_angle = self.dec_angle(self.tetramino_angle)
            self.tetramino = tetraminos[self.tetramino_shape][self.tetramino_angle]
            return
        elif action == Action.ROTATE_RIGHT and self.would_fit(self.inc_angle(self.tetramino_angle)):
            self.tetramino_angle = self.inc_angle(self.tetramino_angle)
            self.tetramino = tetraminos[self.tetramino_shape][self.tetramino_angle]
            return
        else:
            return

    def dec_angle(self, angle):
        new_angle = angle - 1
        if (new_angle < 0):
            new_angle = new_angle + 4
        return new_angle

    def inc_angle(self, angle):
        new_angle = angle + 1
        if (new_angle > 3):
            new_angle = new_angle - 4
        return new_angle

    def has_room_left(self):
        (th, tw) = self.tetramino.shape
        for y in range(th):
            for x in range(tw):
                if (self.tetramino[y][x] == 1 and
                        (self.touches_left(x) or self.clashes_at(x - 1, y))):
                    return False
        return True

    def has_room_right(self):
        (th, tw) = self.tetramino.shape
        for y in range(th):
            for x in range(tw):
                if (self.tetramino[y][x] == 1 and
                        (self.touches_right(x) or self.clashes_at(x + 1, y))):
                    return False
        return True

    def has_room_below(self):
        (th, tw) = self.tetramino.shape
        for y in range(th):
            for x in range(tw):
                if (self.tetramino[y][x] == 1 and
                        (self.on_the_floor(y) or self.clashes_at(x, y + 1))):
                    return False
        return True

    def would_fit(self, new_angle):
        new_tetramino = tetraminos[self.tetramino_shape][new_angle]
        (th, tw) = new_tetramino.shape
        for y in range(th):
            for x in range(tw):
                if (new_tetramino[y][x] == 1 and
                        (self.sticking_out(x, y) or self.clashes_at(x, y))):
                    return False
        return True

    def on_the_floor(self, y):
        return (self.tetramino_y + y == WELL_DEPTH - 1)

    def touches_left(self, x):
        return (self.tetramino_x + x == 0)

    def touches_right(self, x):
        return (self.tetramino_x + x == WELL_WIDTH - 1)

    def sticking_out(self, x, y):
        return (self.tetramino_y + y >= WELL_DEPTH
                or self.tetramino_x + x < 0 or self.tetramino_x + x >= WELL_WIDTH)

    def clashes_at(self, x, y):
        return self.well[self.tetramino_y + y][self.tetramino_x + x] == 1

    def sink(self):
        self.tetramino_y += 1

    def land(self):
        (th, tw) = self.tetramino.shape
        for y in range(th):
            for x in range(tw):
                if (self.tetramino_y + y < WELL_DEPTH and self.tetramino[y][x] == 1):
                    self.well[self.tetramino_y +
                              y][self.tetramino_x + x] = self.tetramino[y][x]

    def emerge(self):
        self.tetramino_shape = self.get_random_shape()
        self.tetramino_angle = 0
        self.tetramino = tetraminos[self.tetramino_shape][self.tetramino_angle]
        self.tetramino_x = TETRAMINO_START_X_POS
        self.tetramino_y = 0

    def is_game_over(self):
        (th, tw) = self.tetramino.shape
        for y in range(th):
            for x in range(tw):
                if (self.tetramino[y][x] == 1 and self.clashes_at(x, y)):
                    return True
        return False

    def draw(self):
        (h, w) = self.well.shape
        (th, tw) = self.tetramino.shape

        def get_tetramino_block(y, x, tetramino):
            if (y - self.tetramino_y >= 0 and y - self.tetramino_y < th and
                x - self.tetramino_x >= 0 and x - self.tetramino_x < tw and
                    tetramino[y - self.tetramino_y][x - self.tetramino_x] == 1):
                return 1
            return 0

        for y in range(h):
            for x in range(w):
                if (self.well[y][x] == 1 or get_tetramino_block(y, x, self.tetramino) == 1):
                    val = '*'
                else:
                    val = '.'
                print(val, end='')
            print()
