# States
# 0/0 -> 0:Init
# 1/0 -> 1
# 2/0 -> 2
# ...
# CONSECUTIVE_WIN / 0 -> CONSECUTIVE_WIN:Terminal
# 0/1 -> CONSECUTIVE_WIN + 1
# 0/2 -> CONSECUTIVE_WIN + 2
# ...
# 0/CONSECUTIVE_WIN -> CONSECUTIVE_WIN * 2:Terminal

# Actions
# 0 for Rock
# 1 for Paper
# 2 for Scissor

CONSECUTIVE_WIN = 4


class RPSEnv:
    def __init__(self):
        pass

    def get_name(self):
        return 'sRPS'

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action_pl, action_op):
        if self.current_state == self.p1_win_terminal_state():
            return None, 10, True, 'player wins'
        elif self.current_state == self.p2_win_terminal_state():
            return None, -10, True, 'opponent wins'
        res = self.get_action_result(action_pl, action_op)
        if res == 0:  # draw
            pass  # do nothing
        elif res == 1:  # pl win
            if self.current_state < self.p1_win_terminal_state():
                self.current_state += 1
            else:
                self.current_state = 1
        else:  # op win
            if self.p1_win_terminal_state() < self.current_state < self.p2_win_terminal_state():
                self.current_state += 1
            else:
                self.current_state = self.p1_win_terminal_state() + 1
        return self.current_state, 0, False, ''

    def p1_win_terminal_state(self):
        return CONSECUTIVE_WIN

    def p2_win_terminal_state(self):
        return CONSECUTIVE_WIN * 2

    def get_n_states(self):
        return CONSECUTIVE_WIN * 2 + 1

    def get_n_actions(self):
        return 3

    def is_terminal_state(self, state):
        return state == self.p1_win_terminal_state() or state == self.p2_win_terminal_state()

    def is_non_terminal_state(self, state):
        return not self.is_terminal_state(state)

    def get_action_result(self, action1, action2):
        '''
        Result for player 1 do action 1 and player 2 do action 2
        :param action1: player 1's action
        :param action2: player 2's action
        :return: result of the game, 0 for draw, 1 for p1 win, -1 for p2 win
        '''
        if action1 == 0:
            if action2 == 0:
                return 0
            if action2 == 1:
                return -1
            if action2 == 2:
                return 1
        if action1 == 1:
            if action2 == 0:
                return 1
            if action2 == 1:
                return 0
            if action2 == 2:
                return -1
        if action1 == 2:
            if action2 == 0:
                return -1
            if action2 == 1:
                return 1
            if action2 == 2:
                return 0
