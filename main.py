import numpy as np
import time
import random
import math
from game import Game, Action, WELL_DEPTH, WELL_WIDTH
from network import NeuralNetwork
from experience_buffer import ExperienceBuffer

# todo
import os
def clear(): return os.system('cls')


actions = [Action.LEFT, Action.RIGHT, Action.ROTATE_LEFT,
           Action.ROTATE_RIGHT, Action.DO_NOTHING]

STATE_SIZE = WELL_DEPTH * WELL_WIDTH

EPISODES = 1000
NON_TERMINAL_STATE_CAPACITY = 100
TERMINAL_STATE_CAPACITY = 10

TRAINING_SIZE = 10
TESTING_SIZE = 2

GAMMA = 0.9

if __name__ == "__main__":

    exp_buffer = ExperienceBuffer(
        NON_TERMINAL_STATE_CAPACITY, TERMINAL_STATE_CAPACITY, STATE_SIZE)

    nn = NeuralNetwork(STATE_SIZE)

    def get_action(state):
        action_values = nn.forward(state)
        # print(action_values)
        return np.argmax(action_values)
        # implement epsilon-greedy
        # return random.randint(0, 4)

    for e in range(EPISODES):
        game = Game()
        print('.', end='', flush=True)

        while not game.has_terminated():
            # clear()

            current_state = game.get_state()
            action_idx = get_action(current_state)
            action = actions[action_idx]
            reward = game.next(action)
            next_state = game.get_state()

            exp_buffer.save_sample(
                current_state, action_idx, reward, next_state, game.has_terminated())

            # print(current_state)
            # game.draw()
            # time.sleep(0.1)

            if (exp_buffer.get_samples_total() > TRAINING_SIZE + TESTING_SIZE):
                (train_data, test_data) = exp_buffer.get_train_test(
                    TRAINING_SIZE, TESTING_SIZE)

                # TODO: loop
                sample = train_data[0]
                (state, action_idx, reward, next_state,
                    is_terminal) = exp_buffer.extract_sample_data(sample)

                target = reward
                if not is_terminal:
                    next_state_action_values = nn.forward(
                        next_state)  # todo: use copy
                    target += GAMMA * np.max(next_state_action_values)

                action_values = nn.forward(state)
                targets = np.copy(action_values)
                targets[np.argmax(action_values)] = target

                nn.backprop(targets)

        print(f'Game score: {game.score}')
