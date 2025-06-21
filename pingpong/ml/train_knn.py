import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def load_data(file_path, side):
    data = []
    with open(file_path, "rb") as f:
        try:
            while True:
                batch = pickle.load(f)
                for r in batch:
                    r["side"] = side
                    data.append(r)
        except EOFError:
            pass
    return data

# === Load 1P and 2P data ===
data_1p = load_data("pingpong_data/1P_data.pickle", "1P")
data_2p = load_data("pingpong_data/2P_data.pickle", "2P")
all_data = data_1p + data_2p

# === Convert 2P data to 1P view ===
for r in all_data:
    if r["side"] == "2P":
        r["ball_y"] = 500 - r["ball_y"]
        r["ball_dy"] = -r["ball_dy"]
        r["blocker_y"] = 500 - r.get("blocker_y", 0)
        r["side"] = "1P_mirrored"

# === Count rally rounds ===
rounds = 0
prev_frame = -1
for r in all_data:
    if prev_frame != -1 and r["frame"] < prev_frame:
        rounds += 1
    prev_frame = r["frame"]
rounds += 1
print(f"🔁 Total rallies: {rounds}")

# === Split into individual sessions ===
sessions = []
session = []

for i in range(len(all_data) - 1):
    curr = all_data[i]
    next_ = all_data[i + 1]
    session.append(curr)
    if next_["frame"] <= curr["frame"]:
        sessions.append(session)
        session = []

if session:
    sessions.append(session)

print(f"🎯 Total sessions: {len(sessions)}")

# === Keep only sessions where the current side wins ===
def is_successful_session(session):
    if not session:
        return False
    last = session[-1]
    status = last.get("status", "")
    side = last.get("side", "")

    if status == "GAME_2P_WIN" and side in ["2P", "1P_mirrored"]:
        return True
    if status == "GAME_1P_WIN" and side == "1P":
        return True
    return False

# === Filter only successful sessions ===
successful_sessions = [s for s in sessions if is_successful_session(s)]
successful_data = [r for s in successful_sessions for r in s]

print(f"✅ Successful sessions kept: {len(successful_sessions)}")
print(f"✅ Successful frames collected: {len(successful_data)}")

# === Prepare training data ===
X = []
y = []

for r in successful_data:
    X.append([
        r["ball_x"],
        r["ball_y"],
        r["ball_dx"],
        r["ball_dy"],
        r["platform_x"],
        r["predicted_x"],
        r.get("blocker_x", 0),
        r.get("blocker_y", 0)
    ])
    y.append(r["command"])

# === Exit if no training data available ===
if len(X) == 0:
    print("❌ No successful data to train on. Please run matches to collect data.")
    exit()

# === Train KNN model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("📊 Model evaluation:")
print(classification_report(y_test, knn.predict(X_test)))

# === Save the model ===
os.makedirs("model", exist_ok=True)
with open("model/knn_model.pickle", "wb") as f:
    pickle.dump(knn, f)

print("✅ Model saved to model/knn_model.pickle")
