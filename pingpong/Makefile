# ====== 參數設定 ======
GAME_DIR := ./
SCRIPT_1P := ./ml/ml_play_template_1P.py
SCRIPT_2P := ./ml/ml_play_template_2P.py
ROUNDS := 92
FRAMES := 512
FRAMESRUN := 1048
DIFFICULTY := HARD
VEL := 6

# ====== 主要指令 ======

run_train:
	@echo "🔁 Running $(ROUNDS) MLGame simulations..."
	@for i in $$(seq 1 $(ROUNDS)); do \
		echo "👉 Run $$i / $(ROUNDS)"; \
		python -m mlgame --nd -f $(FRAMESRUN) -i $(SCRIPT_1P) -i $(SCRIPT_2P) $(GAME_DIR) \
			--difficulty $(DIFFICULTY) --game_over_score 15 --init_vel $(VEL); \
	done
	@echo "✅ Completed all $(ROUNDS) rounds."

run_model:
	@echo "Running model against rule-based"
	python -m mlgame -f 60 -i ./ml/ml_play_model.py -i ./ml/ml_play_template_2P.py  ./ --difficulty HARD --game_over_score 5 --init_vel 6
	@echo "✅ Completed all $(ROUNDS) rounds."

train:
	@echo "🧠 Training KNN model from recorded data..."
	python ml/train_knn.py
	@echo "✅ Model training done."

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf log/ pingpong_data/ __pycache__/
	@echo "✅ Cleaned."

run_and_train:
	@echo "🔁 Running $(ROUNDS) MLGame simulations..."
	@for i in $$(seq 1 $(ROUNDS)); do \
		echo "👉 Run $$i / $(ROUNDS)"; \
		python -m mlgame --nd -f $(FRAMESRUN) -i $(SCRIPT_1P) -i $(SCRIPT_2P) $(GAME_DIR) \
			--difficulty $(DIFFICULTY) --game_over_score 15 --init_vel $(VEL); \
	done
	@echo "✅ Completed all $(ROUNDS) rounds."

	@echo "🧠 Training KNN model from recorded data..."
	python ml/train_knn.py
	@echo "✅ Model training done."

	cp model/knn_model.pickle model/knn_v0.pickle

	@echo "self train start"
	python ml/self_train.py --rounds 10
	@echo "✅ Model self training done."

self_train:
	@echo "self train start"
	python ml/self_train.py --rounds 10
	@echo "✅ Model self training done."

temp:
	python -m mlgame --nd -f 60 \
	-i ./ml/ml_play_model.py \
	-i ./ml/ml_play_model_copy.py \
	./ --difficulty HARD --game_over_score 5 --init_vel 6



# conda activate paia

# python -m mlgame -f 60 -i ./ml/ml_play_template_1P.py -i ./ml/ml_play_template_2P.py  ./ --difficulty HARD --game_over_score 5 --init_vel 6
# python -m mlgame -f 1048 -i ./ml/ml_play_template_1P.py -i ./ml/ml_play_template_2P.py  ./ --difficulty HARD --game_over_score 5 --init_vel 6

# python -m mlgame -f 60 -i ./ml/ml_play_model.py -i ./ml/ml_play_template_2P.py  ./ --difficulty HARD --game_over_score 5 --init_vel 6
# python -m mlgame -f 500 -i ./ml/ml_play_model.py -i ./ml/ml_play_model_copy.py  ./ --difficulty HARD --game_over_score 15 --init_vel 6
# python -m mlgame -f 500 -i ./ml/ml_play_P1_F74111144.py -i ./ml/ml_play_P2_F74111144.py  ./ --difficulty HARD --game_over_score 15 --init_vel 6

# python ml/self_train.py --rounds 10
# make run_and_train > logs/run.txt
# --nd
