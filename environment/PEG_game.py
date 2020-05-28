import numpy as np

from environment import rendering


class PEGGame:
    def __init__(self, peg_env):
        self.peg_env = peg_env
        self.viewer = None
        self.render_geoms = None

    def reset(self):
        self.current_state = self.peg_env.initialize_state()
        return self.current_state

    def get_n_actions(self):
        return self.peg_env.get_n_actions()

    def get_n_states(self):
        return self.peg_env.get_n_states()

    def step(self, action_pl, action_op):
        reward = self.peg_env.get_state_reward(self.current_state)
        if self.peg_env.is_terminal_state(self.current_state):  # if terminal state, return reason
            x1, y1, x2, y2 = self.peg_env.state2rc(self.current_state)
            if (x1, y1) in self.peg_env.evasion_blocks:
                info = 'Player evaded'
            if (x1, y1) == (x2, y2):
                info = 'Player caught by Opponent'
            if (x1, y1) in self.peg_env.crash_blocks:
                info = 'Player crashed'
            if (x2, y2) in self.peg_env.crash_blocks:
                info = 'Opponent crashed'
            return None, reward, True, info
        else:  # non-terminal state
            possible_next_states, p = self.peg_env.gen_runtime_next_state(self.current_state, action_pl, action_op)
            self.current_state = np.random.choice(possible_next_states, p=p)
            return self.current_state, reward, False, None

    def get_name(self):
        return self.peg_env.get_name()

    def render(self):
        def compute_pos_from_rc(r, c):
            return 2 / self.peg_env.n_rows * r + 1 / self.peg_env.n_rows - 1, 2 / self.peg_env.n_rows * c + 1 / self.peg_env.n_rows - 1

        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)
            self.viewer.geoms = []
            self.viewer.set_bounds(-1, 1, -1, 1)
        if self.render_geoms is None:
            self.render_geoms = []
            self.render_geoms_xform = []
            pl = rendering.make_circle(0.07)
            pl_xform = rendering.Transform()
            pl.set_color(0.35, 0.85, 0.35)
            pl.add_attr(pl_xform)
            self.render_geoms.append(pl)
            self.render_geoms_xform.append(pl_xform)
            op = rendering.make_circle(0.07)
            op_xform = rendering.Transform()
            op.set_color(0.85, 0.35, 0.35)
            op.add_attr(op_xform)
            self.render_geoms.append(op)
            self.render_geoms_xform.append(op_xform)
            for crash_block in self.peg_env.crash_blocks:
                block = rendering.make_circle(0.05)
                block_xform = rendering.Transform()
                block.set_color(0.5, 0.5, 0.5)
                block.add_attr(block_xform)
                self.render_geoms.append(block)
                self.render_geoms_xform.append(block_xform)
                self.render_geoms_xform[-1].set_translation(*compute_pos_from_rc(crash_block[0], crash_block[1]))
            for evasion_block in self.peg_env.evasion_blocks:
                block = rendering.make_circle(0.05)
                block_xform = rendering.Transform()
                block.set_color(0.1, 0.1, 0.1)
                block.add_attr(block_xform)
                self.render_geoms.append(block)
                self.render_geoms_xform.append(block_xform)
                self.render_geoms_xform[-1].set_translation(*compute_pos_from_rc(evasion_block[0], evasion_block[1]))

            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

        x1, y1, x2, y2 = self.peg_env.state2rc(self.current_state)
        self.render_geoms_xform[0].set_translation(*compute_pos_from_rc(x1, y1))
        self.render_geoms_xform[1].set_translation(*compute_pos_from_rc(x2, y2))

        results = [self.viewer.render()]
        return results
