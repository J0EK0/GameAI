# Makefile for running 100 episodes of mlgame with rl agents

# 訓練次數（可用 make run COUNT=50 改變）
COUNT ?= 1000

# 遊戲參數
FPS = 500
LEVEL = 9
ROUNDS = 1
AGENT1 = ./ml/ml_play_rl.py
AGENT2 = ./ml/ml_play_rl.py

# 指令模板
GAME_CMD = python -m mlgame --ns --nd -f $(FPS) --one-shot -i $(AGENT1) -i $(AGENT2) . --level $(LEVEL) --game_times $(ROUNDS)

# 執行訓練
temp:
	@echo "🚀 Running $(COUNT) episodes with FPS=$(FPS), LEVEL=$(LEVEL), GAME_TIMES=$(ROUNDS)"
	@for i in $(shell seq 1 $(COUNT)); do \
		echo "======================="; \
		echo "🔁 Episode $$i/$(COUNT)"; \
		$(GAME_CMD); \
	done

train_nd:
	python -m mlgame --ns --nd -f 1000 -i ./ml/ml_play_rl.py -i ./ml/ml_play_rl.py . --level_file levels/contest.json --game_times 1

train:
	python -m mlgame --ns -f 1000 -i ./ml/ml_play_rl.py -i ./ml/ml_play_rl.py . --level_file levels/contest.json --game_times 1

run:
	python -m mlgame --ns -f 500 -i ./ml/ml_play_model_r.py -i ./ml/ml_play_model_r.py . --level_file levels/contest.json --game_times 1
	
graph:
	python ml/graph.py

train_r:
	python -m mlgame --ns --nd -f 1000 -i ./ml/ml_play_rl_rectangle_1.py -i ./ml/ml_play_rl_rectangle_1.py . --level_file levels/contest.json --game_times 1

train_e:
	python -m mlgame --ns --nd -f 1000 -i ./ml/ml_play_rl_enemy.py -i ./ml/ml_play_rl_enemy.py . --level_file levels/contest.json --game_times 1

train_re:
	python -m mlgame --ns --nd -f 1000 -i ./ml/ml_play_rl_enemy.py -i ./ml/ml_play_model_r.py . --level_file levels/contest.json --game_times 1

clean:
	rm -rf ml/model
	mkdir ml/model

# conda activate squid
# python -m mlgame --ns --nd -f 120 -i ./ml/ml_play_manual_1P.py -i ./ml/ml_play_manual_2P.py . --level 8 --game_times 1
# python -m mlgame --ns --nd -f 120 --one-shot -i ./ml/rl_agent.py -i ./ml/ml_play_manual_2P.py . --level 9 --game_times 1

# python -m mlgame --ns -f 120 --one-shot -i ./ml/ml_play_manual_1P.py -i ./ml/ml_play_manual_2P.py . --level_file levels/contest.json --game_times 1

# python -m mlgame --ns -f 500 --one-shot -i ./ml/ml_play_rl.py -i ./ml/ml_play_rl.py . --level 6 --game_times 1
# python -m mlgame --ns --nd -f 500 -i ./ml/ml_play_rl.py -i ./ml/ml_play_rl.py . --level 9 --game_times 1


# ml_play_model_r perform better than all other models