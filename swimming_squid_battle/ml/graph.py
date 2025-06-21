import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取 reward log 檔案（請確認路徑與名稱）
df = pd.read_csv("ml/model_re1/reward_log_1P.csv")  # 修改成你的檔名/路徑

# --- 折線圖 ---
episode_rewards = df.groupby("episode")["reward"].sum()
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards.index, episode_rewards.values, marker="o", linestyle="-", linewidth=1.5)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.grid(True)
plt.tight_layout()
plt.show()

df = pd.read_csv("ml/model_e/reward_log_1P.csv")  # 修改成你的檔名/路徑

# --- 折線圖 ---
episode_rewards = df.groupby("episode")["reward"].sum()
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards.index, episode_rewards.values, marker="o", linestyle="-", linewidth=1.5)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.grid(True)
plt.tight_layout()
plt.show()

"""
# --- Heatmap ---
pivot_df = df.pivot(index="episode", columns="frame", values="reward").fillna(0)
plt.figure(figsize=(14, 6))
sns.heatmap(pivot_df, cmap="RdYlGn", center=0, cbar_kws={"label": "Reward"})
plt.title("Reward Heatmap (Episode vs Frame)")
plt.xlabel("Frame")
plt.ylabel("Episode")
plt.tight_layout()
plt.show()

"""
