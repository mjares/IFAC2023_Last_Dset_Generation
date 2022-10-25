import numpy as np


class TrajectoryDataset:
    def __init__(self):
        self.partition = 0
        self.observations = np.empty(1)
        self.weight = np.empty(1)  # 0 is the loe controller
        self.reward = np.empty(1)
        self.goal = np.empty(1)
        self.safezone_error = np.empty(1)
        self.safe_region = np.empty(1)
        self.trajectories = []  # [x, y, z]
        self.actions = np.empty(1)
        self.conditions = 'Default Conditions'
        self.stable_at_goal = False


class TrajectorySimplifiedDataset:
    def __init__(self):
        self.weight = np.empty(1)  # 0 is the loe controller
        self.estimated_mode = np.empty(1)
        self.safe_region = np.empty(1)
        self.trajectories = []  # [x, y, z]
        self.conditions = 'Default Conditions'
        self.stable_at_goal = False


class HildensiaResults:
    def __init__(self):
        self.dataset = HildensiaDataset()
        self.q_full = 0
        self.p_full = 0
        self.pcp_full = 0
        self.truncation = 0
        self.notes = ""


class HildensiaDataset:
    def __init__(self):
        self.partition = 0
        self.observations = np.empty(1)
        self.weight = np.empty(1)  # 0 is the loe controller
        self.reward = np.empty(1)
        self.goal = np.empty(1)
        self.safezone_error = np.empty(1)
        self.safe_region = np.empty(1)
        self.trajectories = []  # [x, y, z]
        self.actions = np.empty(1)
        self.conditions = 'Default Conditions'
        self.stable_at_goal = False