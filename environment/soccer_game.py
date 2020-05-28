import numpy as np

from environment import rendering


class SoccerGame:
    def __init__(self, soccerEnv):
        self.env = soccerEnv
        self.viewer = None
        self.render_geoms = None
        self.current_state = None

    def get_n_actions(self):
        return self.env.get_n_actions()

    def get_n_states(self):
        return self.env.get_n_states()

    def reset(self):
        self.current_state = np.random.choice(self.env.get_non_terminal_states())
        return self.current_state

    def step(self, action_pl, action_op):
        reward = self.env.get_state_reward(self.current_state)
        if self.env.is_terminal_state(self.current_state):  # if terminal state, return reason
            return None, reward, True, 'Finished'
        else:  # non-terminal state
            self.current_state = self.env.get_runtime_next_state(self.current_state, action_pl, action_op)
            return self.current_state, reward, False, None

    def render(self):
        def compute_pos_from_rc(r, c):
            return  2 / self.env.n_rows * c + 1 / self.env.n_rows - 1, 2 / self.env.n_rows * r + 1 / self.env.n_rows - 1

        if self.viewer is None:
            self.viewer = rendering.Viewer(800, 800)
            self.viewer.geoms = []
            self.viewer.set_bounds(-1, 1, -1, 1)
        if self.render_geoms is None:
            self.render_geoms = []
            self.render_geoms_xform = []
            pl = rendering.make_circle(0.07)
            pl_xform = rendering.Transform()
            pl.set_color(0.0, 0.9, 0.9)
            pl.add_attr(pl_xform)
            self.render_geoms.append(pl)
            self.render_geoms_xform.append(pl_xform)
            op = rendering.make_circle(0.1)
            op_xform = rendering.Transform()
            op.set_color(0.9, 0.0, 0.9)
            op.add_attr(op_xform)
            self.render_geoms.append(op)
            self.render_geoms_xform.append(op_xform)

        x1, y1, x2, y2, ball = self.env._index2rcb(self.current_state)

        self.render_geoms[ball].set_color(0.9, 0.9, 0.9)
        color = [0.9,0.9,0.9]
        color[1-ball] -= 0.9
        self.render_geoms[1-ball].set_color(color[0],color[1],color[2])


        self.render_geoms_xform[0].set_translation(*compute_pos_from_rc(x1, y1))

        self.render_geoms_xform[1].set_translation(*compute_pos_from_rc(x2, y2))

        for geom in self.render_geoms:
            self.viewer.add_geom(geom)
        results = [self.viewer.render()]
        return results
