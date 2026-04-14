from stable_baselines3 import PPO
from swing_env import SwingEnv

# 自作環境のインスタンス化
env = SwingEnv()

# モデルの作成 (MlpPolicy: 64x2層)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# 学習開始
print("ブランコの漕ぎ方を学習中...")
model.learn(total_timesteps=1000000)

# 保存
model.save("ai_swing_model")


# --- 学習済みモデルでテスト走行 ---
print("AIによるテスト走行中...")
test_env = SwingEnv()
log_data = ("episode,t,x,z,dz,phi,d_phi,d2_phi,"
            "alpha,d_alpha,d2_alpha,"
            "torq_grav,torq_ai,torq_ai_head\n")

num_test_episodes = 3  # 何回テスト走行させるか

for episode in range(num_test_episodes):
    obs, _ = test_env.reset()
    print(f"Episode{episode+1} 開始")
    
    for _ in range(2000): # 1エピソードあたりの最大ステップ
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        # データの記録（episode番号付き）
        sw = test_env.sw
        log_data += (f"{episode},{sw.t},{sw.x},{sw.z},{sw.dz},{sw.phi},{sw.d_phi},{sw.d2_phi},"
                     f"{sw.alpha},{sw.d_alpha},{sw.d2_alpha},"
                     f"{sw.torq_grav},{sw.torq_ai},{sw.torq_ai_head}\n")
        
        if terminated or truncated:
            print(f"終了 {'terminated' if terminated else 'truncated'}")
            break # ステップのループを抜けて、次のエピソード（reset）へ

# ファイルに書き出し
with open("ai_swing_result.csv", "w") as f:
    f.write(log_data)
