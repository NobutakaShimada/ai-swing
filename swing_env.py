import gymnasium as gym
from gymnasium import spaces
import numpy as np
from swing_physics import Swing
import math

class SwingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # アクション: [u_body] 腰部トルクのみ (頭部は胴体に固定)
        self.action_space = spaces.Box(low=-500.0, high=500.0, shape=(1,), dtype=np.float32)
        # 状態: [phi, d_phi, alpha, d_alpha, x, z]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.sw = Swing(eps=1, VR=0, x=15, coef_Hooke=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sw.reset()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.sw.phi, self.sw.d_phi,
                         self.sw.alpha, self.sw.d_alpha,
                         self.sw.x, self.sw.z], dtype=np.float32)

    def step(self, action):
        # 1回のAI判断につき、内部物理を20ms分(20ステップ)進める
        u_body = action[0]
        u_head = 0.0  # 頭部は胴体に固定
        for _ in range(20):
            self.sw.rk4_step(u_body, u_head)
            self.sw.t += self.sw.h

        new_obs = self._get_obs()
        # 連続報酬: (1-cos(x))^2 - 大振幅ほど比例以上に報酬が伸びる
        h = 1.0 - math.cos(self.sw.x)
        reward = h * h * 1000.0

        # 終了判定
        terminated = bool(abs(self.sw.x) > math.pi)                          # ブランコ一回転
        terminated = terminated or bool(abs(self.sw.phi - self.sw.x) > 60 * math.pi / 180)  # 胴体のブランコに対する相対角
        terminated = terminated or bool(abs(self.sw.alpha - self.sw.phi) > 60 * math.pi / 180)        # 頭部の胴体に対する相対角
        truncated = bool(self.sw.t > 200.0)

        return new_obs, reward, terminated, truncated, {}
