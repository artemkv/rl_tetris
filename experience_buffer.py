import numpy as np


class ExperienceBuffer:
    def __init__(self, non_terminal_states_capacity, terminal_states_capacity, state_size):
        self.non_term_capacity = non_terminal_states_capacity
        self.term_capacity = terminal_states_capacity

        self.non_term_buffer = np.zeros(
            (self.non_term_capacity, state_size*2 + 3))
        self.term_buffer = np.zeros((self.term_capacity, state_size*2 + 3))

        self.state_size = state_size

        self.non_term_pos = 0
        self.term_pos = 0

        self.non_term_cnt = 0
        self.term_cnt = 0

        self.non_term_loop_over = False
        self.term_loop_over = False

    def save_sample(self, state, action, reward, next_state, is_terminal):
        if is_terminal:
            self.term_buffer[self.term_pos][0:self.state_size] = state
            self.term_buffer[self.term_pos][self.state_size:self.state_size + 1] = action
            self.term_buffer[self.term_pos][self.state_size +
                                            1:self.state_size + 2] = reward
            self.term_buffer[self.term_pos][self.state_size +
                                            2:self.state_size*2 + 2] = next_state
            self.term_buffer[self.term_pos][self.state_size*2 + 2:self.state_size *
                                            2 + 3] = is_terminal
            self.term_pos += 1
            if (self.term_pos == self.term_capacity):
                self.term_pos = 0

                if not self.term_loop_over:
                    self.term_cnt += 1

                self.term_loop_over = True

            if not self.term_loop_over:
                self.term_cnt += 1
        else:
            self.non_term_buffer[self.non_term_pos][0:self.state_size] = state
            self.non_term_buffer[self.non_term_pos][self.state_size:self.state_size + 1] = action
            self.non_term_buffer[self.non_term_pos][self.state_size +
                                                    1:self.state_size + 2] = reward
            self.non_term_buffer[self.non_term_pos][self.state_size +
                                                    2:self.state_size*2 + 2] = next_state
            self.non_term_buffer[self.non_term_pos][self.state_size*2 + 2:self.state_size *
                                                    2 + 3] = is_terminal
            self.non_term_pos += 1
            if (self.non_term_pos == self.non_term_capacity):
                self.non_term_pos = 0

                if not self.non_term_loop_over:
                    self.non_term_cnt += 1

                self.non_term_loop_over = True

            if not self.non_term_loop_over:
                self.non_term_cnt += 1

    def get_samples_total(self):
        return self.non_term_cnt + self.term_cnt

    def get_train_test(self, train_size, test_size):
        data = np.concatenate(
            (self.non_term_buffer[:self.non_term_cnt], self.term_buffer[:self.term_cnt]), axis=0)
        np.random.shuffle(data)
        data = data[:train_size+test_size]
        train_data = data[:train_size]
        test_data = data[train_size:]
        return (train_data, test_data)

    def extract_sample_data(self, sample):
        return (sample[0:self.state_size],
                sample[self.state_size:self.state_size + 1][0],
                sample[self.state_size + 1:self.state_size + 2][0],
                sample[self.state_size + 2:self.state_size * 2+2],
                sample[self.state_size*2 + 2:self.state_size * 2 + 3][0])
