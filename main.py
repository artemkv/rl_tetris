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

EPISODES = 10000000
NON_TERMINAL_STATE_CAPACITY = 900
TERMINAL_STATE_CAPACITY = 100

TRAINING_SIZE = 100
TESTING_SIZE = 1

GAMMA = 0.9

SWAP_AFTER_EPISODES = 10

EPSILON = 0.1


def loss(y, t):
    return (t - y) * (t - y)


if __name__ == "__main__":

    exp_buffer = ExperienceBuffer(
        NON_TERMINAL_STATE_CAPACITY, TERMINAL_STATE_CAPACITY, STATE_SIZE)

    nn = NeuralNetwork(STATE_SIZE)
    nn1 = NeuralNetwork(STATE_SIZE)

    def get_action(state):
        p = np.random.random()
        if p < EPSILON:
            return random.randint(0, 4)
        else:
            action_values = nn.forward(state)
            return np.argmax(action_values)

    for e in range(EPISODES):
        game = Game()

        while not game.has_terminated():
            # clear()
            print('.', end='', flush=True)

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

                total_loss = 0
                while True:
                    prev_loss = total_loss
                    total_loss = 0

                    for sample in train_data:
                        (state, action_idx, reward, next_state,
                            is_terminal) = exp_buffer.extract_sample_data(sample)

                        target = reward
                        if not is_terminal:
                            next_state_action_values = nn1.forward(next_state)
                            target += GAMMA * np.max(next_state_action_values)

                        action_values = nn.forward(state)
                        targets = np.copy(action_values)
                        targets[np.argmax(action_values)] = target

                        # print(                            f'predicted: {np.max(action_values)}, target: {target}')

                        sample_loss = loss(np.max(action_values), target)

                        nn.backprop(targets)

                        new_action_values = nn.forward(state)
                        new_sample_loss = loss(
                            new_action_values[np.argmax(action_values)], target)

                        # print(                            f'before: {sample_loss}, after: {new_sample_loss}')

                        total_loss += sample_loss

                    total_loss = total_loss / TRAINING_SIZE
                    if abs(total_loss - prev_loss) < 0.001:
                        # print(f'Loss: {total_loss}')
                        break

        print(f'Game score: {game.score}')

        # if (e % SWAP_AFTER_EPISODES):
        #    nn1 = nn.copy()
