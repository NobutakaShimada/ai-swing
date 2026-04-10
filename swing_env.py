import gymnasium as gym
from gymnasium import spaces
import numpy as np
from swing_physics import Swing
import math

class SwingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # アクション: トルク (-60 ～ +200)
        self.action_space = spaces.Box(low=-500.0, high=500.0, shape=(1,), dtype=np.float32)
        # 状態: [phi, d_phi, x, z]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.sw = Swing(eps=1,VR=1,x=5,coef_Hooke=0) #####ここがポイント######

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sw.reset()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.sw.phi, self.sw.d_phi, self.sw.x, self.sw.z], dtype=np.float32)

    def step(self, action):
        old_energy = self.sw.observe()
        
        # 1回のAI判断につき、内部物理を20ms分(20ステップ)進める
        u = action[0]
        for _ in range(20):
            self.sw.LB_rk4(u)
            self.sw.SW_rk4()
            self.sw.t += self.sw.h

        new_obs = self._get_obs()
        new_energy = self.sw.observe()


        # 報酬: エネルギーの増分 
        reward = (new_energy - old_energy) * 60 
        if self.sw.phi-self.sw.delta_phi_0 < 40*math.pi/180 : #上体角振幅40°以下
            reward +=10


        # 終了判定: 200秒経過するか、回転したら終了
        terminated = bool(abs(self.sw.x) > math.pi)
        terminated = bool(abs(self.sw.phi-self.sw.delta_phi_0) > 65*math.pi/180)
        truncated = bool(self.sw.t > 200.0)          #↑上体角大きすぎたら止め

        return new_obs, reward, terminated, truncated, {}
