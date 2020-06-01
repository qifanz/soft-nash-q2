from math import floor

import numpy as np

TERMINAL_BLOCKS = [[(1, 0), (2, 0)], [(1, 4), (2, 4)]]
REWARD = 10
OFFSET_REWARD = -0.1


class SoccerEnv:
    def __init__(self):
        self.n_rows = 4
        self.n_cols = 5
        self.n_actions = 5
        self.states, self.terminal_states, self.non_terminal_states, self.rewards = self._init_states()
        self.transitions = self._init_transitions()

    def get_name(self):
        return 'soccer'

    def is_terminal_state(self, state):
        return state in self.terminal_states

    def is_non_terminal_state(self, state):
        return state in self.non_terminal_states

    def get_n_states(self):
        return len(self.states)

    def get_n_actions(self):
        return self.n_actions

    def get_state_reward(self, state):
        return self.rewards[state]

    def get_terminal_states(self):
        return self.terminal_states

    def get_non_terminal_states(self):
        return self.non_terminal_states

    def get_state_transition(self, action_pair):
        '''
        Get the transition matrix for pair of actions
        :param action_pair: tuple of actions (action of player1, action of player 2)
        :return: transition matrix given that pair of action
        '''
        return self.transitions[(action_pair[0], action_pair[1])]

    def get_action_str(self, action):
        if action == 0:
            return 'Down'
        elif action == 1:
            return 'Left'
        elif action == 2:
            return 'Up'
        elif action == 3:
            return 'Right'
        elif action == 4:
            return 'Stay'

    def get_action_movement(self, action):
        assert (action < self.n_actions)
        if action == 0:
            return -1, 0
        elif action == 1:
            return 0, -1
        elif action == 2:
            return 1, 0
        elif action == 3:
            return 0, 1
        else:
            return 0, 0

    def get_next_state_1(self, state, action1, action2):
        '''
        player 1 moves first
        '''
        stay1 = self.get_action_str(action1) == 'Stay'
        stay2 = self.get_action_str(action2) == 'Stay'
        r1, c1, r2, c2, ball = self._index2rcb(state)
        new_r1, new_c1 = r1 + self.get_action_movement(action1)[0], c1 + self.get_action_movement(action1)[
            1]
        new_r2, new_c2 = r2 + self.get_action_movement(action2)[0], c2 + self.get_action_movement(action2)[
            1]
        if not self.is_action_valid(r1, c1, action1):
            stay1 = True
            new_r1, new_c1 = r1, c1
        if not self.is_action_valid(r2, c2, action2):
            stay2 = True
            new_r2, new_c2 = r2, c2
        # 1. case player 2 stays, player 1 moves toward player 2
        # player 1 bounces back and ball switches to player 2
        if stay2 and new_r1 == new_r2 and new_c1 == new_c2:
            new_r1 = r1
            new_c1 = c1
            ball = 1
            assert (not stay1)  # otherwise theres a problem here
        # 2. case player 1 stays, player 2 moves toward player 1
        elif stay1 and new_r1 == new_r2 and new_c1 == new_c2:
            new_r2 = r2
            new_c2 = c2
            ball = 0
            assert (not stay2)
        # 3. case player 1 and player 2 move toward same block
        elif new_r1 == new_r2 and new_c1 == new_c2:
            new_r2 = r2
            new_c2 = c2
            ball = 0
        return self._rcb2index(new_r1, new_c1, new_r2, new_c2, ball)

    def get_next_state_2(self, state, action1, action2):
        '''
        player 2 moves first
        '''
        stay1 = self.get_action_str(action1) == 'Stay'
        stay2 = self.get_action_str(action2) == 'Stay'
        r1, c1, r2, c2, ball = self._index2rcb(state)
        new_r1, new_c1 = r1 + self.get_action_movement(action1)[0], c1 + self.get_action_movement(action1)[
            1]
        new_r2, new_c2 = r2 + self.get_action_movement(action2)[0], c2 + self.get_action_movement(action2)[
            1]
        if not self.is_action_valid(r1, c1, action1):
            stay1 = True
            new_r1, new_c1 = r1, c1
        if not self.is_action_valid(r2, c2, action2):
            stay2 = True
            new_r2, new_c2 = r2, c2
        # 1. case player 2 stays, player 1 moves toward player 2
        # player 1 bounces back and ball switches to player 2
        if stay2 and new_r1 == new_r2 and new_c1 == new_c2:
            new_r1 = r1
            new_c1 = c1
            ball = 1
            assert (not stay1)  # otherwise theres a problem here
        # 2. case player 1 stays, player 2 moves toward player 1
        elif stay1 and new_r1 == new_r2 and new_c1 == new_c2:
            new_r2 = r2
            new_c2 = c2
            ball = 0
            assert (not stay2)
        # 3. case player 1 and player 2 move toward same block
        elif new_r1 == new_r2 and new_c1 == new_c2:
            new_r1 = r1
            new_c1 = c1
            ball = 1
        return self._rcb2index(new_r1, new_c1, new_r2, new_c2, ball)

    def get_runtime_next_state(self, state, action1, action2):
        return np.random.choice(self.get_next_states(state, action1, action2))

    def get_next_states(self, state, action1, action2):
        if state in self.non_terminal_states:
            new_state1 = self.get_next_state_1(state, action1, action2)
            new_state2 = self.get_next_state_2(state, action1, action2)
            if new_state1 == new_state2:
                return [new_state1]
            else:
                return [new_state1, new_state2]
        else:
            raise Exception(state, ' is not a non-terminal state')

    def _init_transitions(self):
        transitions = {}
        n_blocks = self.n_rows * self.n_cols
        for action1 in range(self.n_actions):
            for action2 in range(self.n_actions):
                transitions[(action1, action2)] = np.zeros((2 * n_blocks ** 2, 2 * n_blocks ** 2))
                for state in range(2 * n_blocks ** 2):
                    if state in self.non_terminal_states:
                        new_states = self.get_next_states(state, action1, action2)
                        for new_state in new_states:
                            transitions[(action1, action2)][state, new_state] = 1 / len(new_states)
        return transitions

    def is_action_valid(self, row, col, action):
        new_row = row + self.get_action_movement(action)[0]
        new_col = col + self.get_action_movement(action)[1]
        if 0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols:
            return True
        return False

    def _rcb2index(self, row1, col1, row2, col2, ball):
        n_blocks = self.n_rows * self.n_cols
        state1 = row1 * self.n_cols + col1
        state2 = row2 * self.n_cols + col2
        state = state1 * n_blocks + state2
        return state + ball * n_blocks ** 2

    def _index2rcb(self, state):
        n_blocks = self.n_rows * self.n_cols
        ball = int(state / (n_blocks ** 2))
        state %= n_blocks ** 2
        state1 = floor(state / n_blocks)
        state2 = state % n_blocks
        row1 = floor(state1 / self.n_cols)
        col1 = state1 % self.n_cols
        row2 = floor(state2 / self.n_cols)
        col2 = state2 % self.n_cols
        return row1, col1, row2, col2, ball

    def _init_states(self):
        n_blocks = self.n_rows * self.n_cols
        states = np.arange(2 * n_blocks ** 2)
        terminal_states = []
        non_terminal_states = []
        rewards = []
        for i in range(2 * n_blocks ** 2):
            row1, col1, row2, col2, ball = self._index2rcb(i)
            if row1 == row2 and col1 == col2:
                rewards.append(0)
                continue
            else:
                if ball == 0:
                    if (row1, col1) in TERMINAL_BLOCKS[0]:
                        rewards.append(REWARD)
                        terminal_states.append(i)
                    else:
                        rewards.append(OFFSET_REWARD)
                        non_terminal_states.append(i)
                else:
                    if (row2, col2) in TERMINAL_BLOCKS[1]:
                        rewards.append(-REWARD)
                        terminal_states.append(i)
                    else:
                        rewards.append(-OFFSET_REWARD)
                        non_terminal_states.append(i)

        return states, terminal_states, non_terminal_states, rewards


SoccerEnv()
