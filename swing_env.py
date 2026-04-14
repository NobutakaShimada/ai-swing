import gymnasium as gym
from gymnasium import spaces
import numpy as np
from swing_physics import Swing
import math

class SwingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # アクション: [u_body, u_head] 腰部トルクと頭部トルク
        self.action_space = spaces.Box(low=-500.0, high=500.0, shape=(2,), dtype=np.float32)
        # 状態: [phi, d_phi, alpha, d_alpha, x, z]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.sw = Swing(eps=1, VR=0, x=5, coef_Hooke=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sw.reset()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.sw.phi, self.sw.d_phi,
                         self.sw.alpha, self.sw.d_alpha,
                         self.sw.x, self.sw.z], dtype=np.float32)

    def step(self, action):
        old_energy = self.sw.observe()

        # 1回のAI判断につき、内部物理を20ms分(20ステップ)進める
        u_body = action[0]
        u_head = action[1]
        for _ in range(20):
            self.sw.rk4_step(u_body, u_head)
            self.sw.t += self.sw.h

        new_obs = self._get_obs()
        new_energy = self.sw.observe()

        # 報酬: エネルギーの増分
        reward = (new_energy - old_energy) * 60
        if self.sw.phi - self.sw.delta_phi_0 < 40 * math.pi / 180:
            reward += 10

        # 終了判定
        terminated = bool(abs(self.sw.x) > math.pi)                          # ブランコ一回転
        terminated = terminated or bool(abs(self.sw.phi - self.sw.delta_phi_0) > 65 * math.pi / 180)  # 胴体角大きすぎ
        terminated = terminated or bool(abs(self.sw.alpha - self.sw.phi) > 60 * math.pi / 180)        # 頭部の相対角大きすぎ
        truncated = bool(self.sw.t > 200.0)

        return new_obs, reward, terminated, truncated, {}
