import gymnasium as gym
from gymnasium import spaces
import numpy as np
from swing_physics import Swing
import math

class SwingEnv(gym.Env):
    def __init__(self, phi_limit_deg=60, psi_limit_deg=90):
        super().__init__()
        # アクション: [u_body, u_knee] 腰部トルクと膝トルク
        self.action_space = spaces.Box(low=-500.0, high=500.0, shape=(2,), dtype=np.float32)
        # 状態: [phi, d_phi, psi, d_psi, x, z]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.phi_limit = phi_limit_deg * math.pi / 180
        self.psi_limit = psi_limit_deg * math.pi / 180
        self.sw = Swing(eps=1, VR=0, x=15, coef_Hooke=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sw.reset()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.sw.phi, self.sw.d_phi,
                         self.sw.psi, self.sw.d_psi,
                         self.sw.x, self.sw.z], dtype=np.float32)

    def step(self, action):
        # 1回のAI判断につき、内部物理を20ms分(20ステップ)進める
        u_body = action[0]
        u_knee = action[1]
        for _ in range(20):
            self.sw.rk4_step(u_body, u_knee)
            self.sw.t += self.sw.h

        new_obs = self._get_obs()
        # 連続報酬: (1-cos(x))^2 - 大振幅ほど比例以上に報酬が伸びる
        h = 1.0 - math.cos(self.sw.x)
        reward = h * h * 1000.0

        # 終了判定
        terminated = bool(abs(self.sw.x) > 2 * math.pi)                      # ブランコ一回転(360°)で終了
        terminated = terminated or bool(abs(self.sw.phi - self.sw.x) > self.phi_limit)  # 胴体のブランコに対する相対角
        terminated = terminated or bool(abs(self.sw.psi - self.sw.x) > self.psi_limit)  # 下腿のブランコに対する相対角(膝は広めに許容)
        truncated = bool(self.sw.t > 200.0)

        return new_obs, reward, terminated, truncated, {}
