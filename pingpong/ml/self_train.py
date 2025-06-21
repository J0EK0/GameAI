# self_train.py
import os
import pickle
import shutil
import subprocess
import argparse

MODEL_DIR = "model"
DATA_DIR = "pingpong_data"
VERSION_FILE = os.path.join(MODEL_DIR, "version.txt")
TRAIN_SCRIPT = "ml/train_knn.py"
PREDICT_SCRIPT = "ml/ml_play_model.py"

# === 0. Parse command line arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=1, help="Number of self-training rounds")
args = parser.parse_args()

# === 1. Ensure directories exist ===
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# === 2. Get current model version ===
if os.path.exists(VERSION_FILE):
    with open(VERSION_FILE, "r") as f:
        version = int(f.read().strip())
else:
    version = 0

for i in range(args.rounds):
    new_version = version + 1
    model_name = f"knn_v{version}.pickle"
    model_new_name = f"knn_v{new_version}.pickle"
    model_path = os.path.join(MODEL_DIR, model_name)
    model_new_path = os.path.join(MODEL_DIR, model_new_name)

    # === 3. Update predict_knn.py to use the latest model ===
    with open(PREDICT_SCRIPT, "r") as f:
        lines = f.readlines()

    with open(PREDICT_SCRIPT, "w") as f:
        for line in lines:
            if line.strip().startswith("MODEL_PATH"):
                f.write(f'MODEL_PATH = "{model_path}"\n')
            else:
                f.write(line)

    # === 4. Run self-play matches using current model ===
    print(f"\n🎮 Starting self-play for model version {new_version}...\n")

    for match in range(10):
        print(f"Match {match + 1} ...")
        subprocess.run([
            "python", "-m", "mlgame", "--nd",
            "-f", "1048",
            "-i", PREDICT_SCRIPT, "-i", PREDICT_SCRIPT,
            "./", "--difficulty", "HARD", "--game_over_score", "15", "--init_vel", "6"
        ])

    # === 5. Run training script to generate next version ===
    print("\n🧠 Training next generation model...")
    subprocess.run(["python", TRAIN_SCRIPT])

    # === 6. Save model with version name ===
    shutil.copy(os.path.join(MODEL_DIR, "knn_model.pickle"), model_new_path)
    with open(VERSION_FILE, "w") as f:
        f.write(str(new_version))

    print(f"\n✅ Self-training completed. Model version {model_new_name} saved!")

    version += 1
