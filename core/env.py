import gymnasium as gym
import numpy as np
import random

# Định nghĩa tên để Log cho đẹp
SCENARIO_NAMES = {0: "Healthy", 1: "Resource Leak", 2: "Ticking Bomb", 3: "Critical Crash", 4: "Stable Equiv"}
ACTION_NAMES = {0: "+10%", 1: "+5%", 2: "Stay", 3: "-5%", 4: "ROLLBACK"}

class CanaryEnv(gym.Env):
    def __init__(self):
        super(CanaryEnv, self).__init__()
        # State: [Weight, E_canary, E_stable, L_canary, L_stable, CPU, Mem, RPS]
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.weight = 0.05
        self.step_count = 0
        self.scenario = random.randint(0, 4)
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        noise = lambda: np.random.normal(0, 0.01)
        e_stable, l_stable = 0.001, 0.1
        
        if self.scenario == 0: # Healthy
            e_canary, l_canary = 0.001 + noise(), 0.09 + noise()
        elif self.scenario == 1: # Leak
            e_canary, l_canary = 0.001, 0.1 + (self.weight * 2) + (self.step_count * 0.05)
        elif self.scenario == 2: # Bomb
            e_canary = 0.8 / (1 + np.exp(-15 * (self.weight - 0.4))) if self.weight > 0.3 else 0.001
            l_canary = 0.12
        elif self.scenario == 3: # Crash
            e_canary, l_canary = 0.5 + noise(), 0.5 + noise()
        else: # Stable
            e_canary, l_canary = 0.001, 0.1

        cpu = self.weight * 0.8 + noise()
        mem = self.weight * 0.7 + (0.5 if self.scenario == 1 else 0)
        rps = 1000 * self.weight
        
        return np.array([self.weight, e_canary, e_stable, l_canary, l_stable, cpu, mem, rps/1000], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        prev_weight = self.weight

        if action == 0: self.weight += 0.1
        elif action == 1: self.weight += 0.05
        elif action == 2: pass
        elif action == 3: self.weight -= 0.05
        elif action == 4:
            self.weight = 0.0
            self.done = True

        self.weight = np.clip(self.weight, 0.0, 1.0)
        obs = self._get_obs()
        reward = 0
        e_canary, l_canary = obs[1], obs[3]

        if e_canary > 0.05 or l_canary > 0.5:
            reward -= 10
            if action != 4: reward -= 50
        
        if not self.done:
            reward += (self.weight - prev_weight) * 10
            if self.weight >= 1.0:
                if self.scenario == 0: reward += 100
                self.done = True
        
        if self.step_count > 50: self.done = True
        return obs, reward, self.done, False, {}