import copy
import os
from stable_baselines3.common.env_checker import check_env
import gym
from gym import spaces
import numpy as np
import torch
import common
from scoring import score_dataset
from CPC import cpc_code, generate_random


def flatten(l):
    return [item for sublist in l for item in sublist]


# TODO: encourage the **LEAST NUMBER OF STEPS**
class CPCAddCrossEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, params: common.CPCTrialParams, max_cross_edges=170, target_succ_rate=0.99):
        super(CPCAddCrossEnv, self).__init__()
        self.max_cross_edges = max_cross_edges
        self.target_succ_rate = target_succ_rate
        self.params = params

        self.cpc = generate_random.random_cpc(params.n, params.m_x, params.dv_x, params.m_z,
                                              params.dv_z, params.seeds[0], params.seeds[1])
        self.orig_cpc = copy.deepcopy(self.cpc)
        self.check_verts = self.cpc.get_all_check_vertices()

        H, _ = self.cpc.get_tanner_graph()

        # Each action corresponds to adding a check edge
        self.action_space = spaces.MultiDiscrete([
                                                 len(self.check_verts),
                                                 len(self.check_verts),
                                                 ])
        self.last_wer = 0
        self.n_steps = 0
        self.added_edges = []

        self.observation_space = spaces.Box(low=0, high=1,
                                                 shape=(H.shape[0], H.shape[1]), dtype=np.uint8)

    def step(self, action):
        if len(self.added_edges) > self.max_cross_edges:
            H, _ = self.cpc.get_tanner_graph()
            return H, 0, True, {}

        # Adding an edge between one parity check is not allowed
        if action[0] == action[1]:
            H, _ = self.cpc.get_tanner_graph()
            return H, -1, False, {}

        edge = cpc_code.CPCEdge(
            self.check_verts[action[0]],
            self.check_verts[action[1]],
            bit_check=False
        )

        if self.cpc.has_edge(edge):
            H, _ = self.cpc.get_tanner_graph()
            return H, 0, False, {}

        self.cpc.add_edge(
            edge
        )
        self.added_edges.append(edge)
        self.cpc.simplify_virtual_edges()

        H, bit_types = self.cpc.get_tanner_graph()

        p_error = common.calculate_tanner_p_error_depolarizing(
            bit_types, self.params.p_error[0], self.params.p_error[1], self.params.p_error[2])

        wsr = score_dataset.run_decoder_wsr(H, p_error)

        self.last_wer = 1 - wsr

        # TODO: SCALED?
        reward = wsr
        self.n_steps += 1
        obs = H

        return obs, reward, wsr >= self.target_succ_rate, {}

    def reset(self):
        # TODO: THIS DOES NOT REMOVE VIRTUAL EDGES... need to store copy of og
        # for e in self.added_edges:
        #     self.cpc.remove_edge(e)
        # self.cpc.simplify_virtual_edges()
        self.cpc = copy.deepcopy(self.orig_cpc)
        H, _ = self.cpc.get_tanner_graph()
        self.added_edges = []
        return H

    def render(self, mode='console'):
        pass

    def close(self):
        pass
