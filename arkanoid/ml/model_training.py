import pickle
import numpy as np
import os
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# **Set directories**
PICKLE_DIR = "pickles_wallandbrick/"
MODEL_PATH = "pickles_model/temp.pickle"
samples_per_level = 50  # **Limit to 50 samples per level to prevent a single level from dominating**

# **Initialize training data**
X, y = [], []
level_data = defaultdict(list)  # **Dictionary to store data for each level**

# **Read all level data**
for filename in os.listdir(PICKLE_DIR):
    if filename.startswith("game_data_levels_") and filename.endswith(".pickle"):
        file_path = os.path.join(PICKLE_DIR, filename)
        try:
            with open(file_path, "rb") as f:
                game_data = pickle.load(f)

                for data in game_data:
                    # **Filtering condition: Keep only successful data**
                    if data["game_state"] == "GAME_PASS" or data["remain_brick"] == 1:
                        feature_vector = [
                            data["ball_x"], data["ball_y"], 
                            data["delta_x"], data["delta_y"], 
                            data["platform_x"], data["predicted_x"]
                        ]
                        level_data[data["level"]].append((feature_vector, data["command"]))

        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            print(f"⚠️ Unable to read {file_path}, skipping")

# **Balance data: Ensure a consistent amount of data per level**
for level, samples in level_data.items():
    if len(samples) > samples_per_level:
        samples = random.sample(samples, samples_per_level)  # **Randomly select a fixed number of samples**
    
    for x_data, y_data in samples:
        X.append(x_data)
        y.append(y_data)

# **Convert to NumPy arrays**
X = np.array(X)
y = np.array(y)

# **Ensure sufficient data**
if len(X) == 0 or len(y) == 0:
    print("⚠️ No valid data available for training. Please check the `pickles/` directory.")
    exit()

# **Split training and test data**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Train the KNN model**
knn = KNeighborsClassifier(n_neighbors=1)  # Set K=3
knn.fit(X_train, y_train)

# **Evaluate model accuracy**
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy * 100:.2f}%")

# **Ensure the target directory exists**
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# **Save the trained KNN model**
with open(MODEL_PATH, "wb") as f:
    pickle.dump(knn, f)

print(f"✅ KNN model training completed! Model saved to {MODEL_PATH}")
