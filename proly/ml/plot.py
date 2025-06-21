import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV
df = pd.read_csv("log/player1_metrics.csv")

# 畫圖
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ["policy_loss", "value_loss", "entropy", "total_loss", "avg_reward"]
titles = ["Policy Loss", "Value Loss", "Entropy", "Total Loss", "Average Reward"]

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i // 3, i % 3]
    ax.plot(df["step"], df[metric])
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.grid(True)

# episode 單獨畫
axes[1, 2].plot(df["step"], df["episode"], color='gray')
axes[1, 2].set_title("Episode Count")
axes[1, 2].set_xlabel("Step")
axes[1, 2].set_ylabel("Episode")
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()
