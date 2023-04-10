import numpy as np
import time
import random
from game import Game, Action, WELL_DEPTH, WELL_WIDTH

# todo
import os
def clear(): return os.system('cls')


actions = [Action.LEFT, Action.RIGHT, Action.ROTATE_LEFT,
           Action.ROTATE_RIGHT, Action.DO_NOTHING]

EXPERIENCE_SIZE = 100000
TERMINAL_STATES = 10000
NON_TERMINALE_STATES = EXPERIENCE_SIZE - TERMINAL_STATES

STATE_SIZE = WELL_DEPTH * WELL_WIDTH

if __name__ == "__main__":

    exp_buffer = np.zeros((EXPERIENCE_SIZE, STATE_SIZE*2 + 2))

    cnt = 0
    terminal_cnt = 0
    non_terminal_cnt = 0
    games_played = 0

    while cnt < EXPERIENCE_SIZE:
        game = Game()
        games_played += 1

        print('.', end='', flush=True)

        while not game.has_terminated() and cnt < EXPERIENCE_SIZE:
            current_state = game.get_state()
            action_idx = random.randint(0, 4)
            action = actions[action_idx]
            reward = game.next(action)
            next_state = game.get_state()

            to_be_saved = False
            if game.has_terminated():
                if terminal_cnt < TERMINAL_STATES:
                    to_be_saved = True
                    terminal_cnt += 1
            else:
                if non_terminal_cnt < NON_TERMINALE_STATES:
                    to_be_saved = True
                    non_terminal_cnt += 1

            if to_be_saved:
                exp_buffer[cnt][0:STATE_SIZE] = current_state
                exp_buffer[cnt][STATE_SIZE:STATE_SIZE + 1] = action_idx
                exp_buffer[cnt][STATE_SIZE + 1:STATE_SIZE+2] = reward
                exp_buffer[cnt][STATE_SIZE+2:STATE_SIZE*2+3] = next_state
                cnt += 1

            # clear()
            # print(current_state)
            # game.draw()
            # time.sleep(0.1)

    print(f'Games played: {games_played}')

    SAMPLE_NO = 400
    print(exp_buffer[SAMPLE_NO][0:STATE_SIZE].reshape(
        (WELL_DEPTH, WELL_WIDTH)))
    print(exp_buffer[SAMPLE_NO][STATE_SIZE:STATE_SIZE + 1])
    print(exp_buffer[SAMPLE_NO][STATE_SIZE+2:STATE_SIZE *
          2+3].reshape((WELL_DEPTH, WELL_WIDTH)))
