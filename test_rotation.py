"""学習済みモデルで360°回転を記録するテスト走行スクリプト"""
from stable_baselines3 import PPO
from swing_env import SwingEnv
import math

model = PPO.load("ai_swing_model")

num_episodes = 10
all_logs = []  # [(episode_idx, log_str, rotation_time)]

for episode in range(num_episodes):
    env = SwingEnv(phi_limit_deg=90, psi_limit_deg=120)
    obs, _ = env.reset()
    # 完全静止スタートを強制
    env.sw.x = 0.0
    env.sw.z = 0.0       # dx
    env.sw.phi = 0.0
    env.sw.d_phi = 0.0
    env.sw.psi = 0.0
    env.sw.d_psi = 0.0
    env.sw.u_body_filt = 0.0
    env.sw.u_knee_filt = 0.0
    obs = env._get_obs()
    log_lines = []
    rotation_time = None

    for step in range(20000):  # 最大400秒分
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        sw = env.sw
        log_lines.append(
            f"{episode},{sw.t},{sw.x},{sw.z},{sw.dz},{sw.phi},{sw.d_phi},{sw.d2_phi},"
            f"{sw.psi},{sw.d_psi},{sw.d2_psi},"
            f"{sw.torq_grav},{sw.torq_ai},{sw.torq_ai_knee}"
        )

        # 360°回転を検出
        if rotation_time is None and abs(sw.x) > 2 * math.pi:
            rotation_time = sw.t
            print(f"Episode {episode}: 360° rotation at t={rotation_time:.1f}s")

        if terminated or truncated:
            break

    all_logs.append((episode, log_lines, rotation_time))
    rt_str = f"{rotation_time:.1f}s" if rotation_time else "未達"
    print(f"Episode {episode}: 終了 (rotation: {rt_str}, steps: {len(log_lines)})")

# 一番早く回転したエピソードを選ぶ
rotated = [(i, lines, rt) for i, lines, rt in all_logs if rt is not None]
if not rotated:
    print("360°回転に達したエピソードがありません")
    # 一番振幅が大きいものを選ぶ
    best = all_logs[0]
else:
    best = min(rotated, key=lambda x: x[2])
    print(f"\n最速回転: Episode {best[0]} at t={best[2]:.1f}s")

# 最速エピソードだけCSV出力 (episode番号を0にリセット)
header = "episode,t,x,z,dz,phi,d_phi,d2_phi,psi,d_psi,d2_psi,torq_grav,torq_ai,torq_ai_knee\n"
with open("ai_swing_rotation.csv", "w") as f:
    f.write(header)
    for line in best[1]:
        # episode番号を0にリセット
        f.write("0," + line.split(",", 1)[1] + "\n")

print(f"ai_swing_rotation.csv に書き出し完了 ({len(best[1])} frames)")
