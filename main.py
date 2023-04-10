import time
import random
from game import Game, Action

# todo
import os
def clear(): return os.system('cls')


actions = [Action.LEFT, Action.RIGHT, Action.ROTATE_LEFT,
           Action.ROTATE_RIGHT, Action.DO_NOTHING]

if __name__ == "__main__":
    game = Game()
    while not game.is_terminated():
        clear()
        game.draw()
        time.sleep(0.01)
        reward = game.next(actions[random.randint(0, 4)])
        state = game.get_state()
