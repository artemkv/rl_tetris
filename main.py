import numpy as np
import time
import random
from game import Game, Action, WELL_DEPTH, WELL_WIDTH
from network import NeuralNetwork
from experience_buffer import ExperienceBuffer

# todo
import os
def clear(): return os.system('cls')


actions = [Action.LEFT, Action.RIGHT, Action.ROTATE_LEFT,
           Action.ROTATE_RIGHT, Action.DO_NOTHING]

STATE_SIZE = WELL_DEPTH * WELL_WIDTH

EPISODES = 1000000
NON_TERMINAL_STATE_CAPACITY = 2000
TERMINAL_STATE_CAPACITY = 1000

TRAINING_SIZE = 50
TESTING_SIZE = 1

GAMMA = 0.9

SWAP_AFTER_EPISODES = 100

EPSILON = 0.001  # 0.001  # 0.2
RANDOM_STREAK = 3

on_random_streak = False
random_streak_cnt = 0


TRAIN_LOSS = 0.01


def loss(y, t):
    return (t - y) * (t - y)


def argmax_with_random_tie_break(aa):
    max = aa.max()
    return np.random.choice(np.where(aa == max)[0])


def get_action(nn, state):
    global on_random_streak
    global random_streak_cnt

    p = np.random.random()
    if not on_random_streak and p < EPSILON:
        random_streak_cnt = 0
        on_random_streak = True
        print('???')

    if on_random_streak:
        action = random.randint(0, 4)
        # print('???')
        random_streak_cnt += 1
        if random_streak_cnt == RANDOM_STREAK:
            on_random_streak = False
        return action
    else:
        action_values = nn.forward(state)
        return np.argmax(action_values)


def get_action_1(nn, state):
    p = np.random.random()
    if p < EPSILON:
        return random.randint(0, 4)
    else:
        action_values = nn.forward(state)
        # print(action_values)
        # return argmax_with_random_tie_break(action_values)
        return np.argmax(action_values)


def get_action_2(nn, state):
    action_values = nn.forward(state)
    return np.argmax(action_values)


def play(nn):
    game = Game()
    while not game.has_terminated():
        clear()
        game.draw()
        time.sleep(0.02)
        action = actions[get_action(nn, game.get_state())]
        print(action)
        game.next(action)


def train(nn, episodes):
    exp_buffer = ExperienceBuffer(
        NON_TERMINAL_STATE_CAPACITY, TERMINAL_STATE_CAPACITY, STATE_SIZE)

    nn1 = nn.copy()

    for e in range(episodes):
        game = Game()

        while not game.has_terminated():
            # clear()
            # print('.', end='', flush=True)

            current_state = game.get_state()
            action_idx = get_action(nn, current_state)
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
                        targets[int(action_idx)] = target

                        # print(f'predicted: {np.max(action_values)}, target: {target}')

                        sample_loss = loss(
                            action_values[int(action_idx)], target)

                        nn.backprop(targets)

                        # new_action_values = nn.forward(state)
                        # new_sample_loss = loss(new_action_values[int(action_idx)], target)

                        # print(f'before: {sample_loss}, after: {new_sample_loss}')

                        total_loss += sample_loss

                    total_loss = total_loss / TRAINING_SIZE
                    if abs(total_loss - prev_loss) < TRAIN_LOSS:
                        # print(f'Loss: {total_loss}')
                        break

        visual_score = ''
        for _ in range(game.score // 10):
            visual_score += '.'
        print(f'E_{e:05d} {visual_score}')

        if (e > 0 and e % SWAP_AFTER_EPISODES == 0):
            nn.save(f'_trained_{e}')
            nn1 = nn.copy()

    nn.save(f'_trained_{EPISODES}')


if __name__ == "__main__":
    nn = NeuralNetwork(STATE_SIZE)
    # nn.load("_trained_3000")
    # nn.save("_initial")
    # train(nn, EPISODES)

    for i in range(100):
        play(nn)
