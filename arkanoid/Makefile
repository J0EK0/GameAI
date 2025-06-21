# Define variables
PYTHON_CMD=python -m mlgame
SCRIPT=./ml/ml_play_brickandwall.py
SCRIPTC=./ml/ml_play_wallonly.py
SCRIPT_MODEL=./ml/ml_play_model.py
FLAGS=-f 2048 -i $(SCRIPT) .
FLAGSC=-f 2048 --one-shot -i $(SCRIPTC) .
FLAGS_TEST=-f 1024 --one-shot -i $(SCRIPT) .
FLAGS_MODEL=-f 1024 --one-shot -i $(SCRIPT_MODEL) .
LOG_DIR=logs
RUNS=1
RUN_TEST=1
LEVELS=$(shell seq 1 24)
LEVELS_TEST=$(shell seq 10 24)
TRAIN_LOG=$(LOG_DIR)/train_log.txt  # training log
TEST_LOG=$(LOG_DIR)/test_log.txt    # test log
MODEL_TEST_LOG=$(LOG_DIR)/model_test_log.txt  # model

# Ensure the logs directory exists
$(shell mkdir -p $(LOG_DIR))

# Default target
all: train run_train

# Train each level multiple times and save all logs in a single file
run_train:
	@echo "🔥 Starting Training for All Levels (Saving output to $(TRAIN_LOG))" > $(TRAIN_LOG)
	@for level in $(LEVELS); do \
		for run in $(shell seq 1 $(RUNS)); do \
			echo "🔥 Running Training Level $$level - Attempt $$run" | tee -a $(TRAIN_LOG); \
			$(PYTHON_CMD) $(FLAGS) --difficulty NORMAL --level $$level >> $(TRAIN_LOG) 2>&1; \
		done; \
	done

train:
	python ./ml/model_training.py

# Test each level and save all logs in a single file
run_test:
	@echo "🔥 Starting AI Test for All Levels (Saving output to $(TEST_LOG))" > $(TEST_LOG)
	@for level in $(LEVELS_TEST); do \
		for run in $(shell seq 1 $(RUN_TEST)); do \
			echo "🔥 Running Test Level $$level - Attempt $$run" | tee -a $(TEST_LOG); \
			$(PYTHON_CMD) $(FLAGS_TEST) --difficulty NORMAL --level $$level >> $(TEST_LOG) 2>&1; \
		done; \
	done

# Test trained model on each level and save logs in a single file
test:
	@echo "🔥 Starting Model Test for All Levels (Saving output to $(MODEL_TEST_LOG))" > $(MODEL_TEST_LOG)
	@for level in $(LEVELS); do \
		for run in $(shell seq 1 $(RUN_TEST)); do \
			echo "🔥 Running Model Test Level $$level - Attempt $$run" | tee -a $(MODEL_TEST_LOG); \
			$(PYTHON_CMD) $(FLAGS_MODEL) --difficulty NORMAL --level $$level >> $(MODEL_TEST_LOG) 2>&1; \
		done; \
	done

# conda activate paia

# python -m mlgame -f 120 -i ./ml/ml_play_brickandwall.py . --difficulty NORMAL --level 5
# python -m mlgame -f 480 -i ./ml/ml_play_brickandwall.py . --difficulty NORMAL --level 5
# python -m mlgame -f 2048 -i ./ml/ml_play_brickandwall.py . --difficulty NORMAL --level 1

# python -m mlgame -f 480 -i ./ml/ml_play_manual.py . --difficulty NORMAL --level 5

# python -m mlgame -f 1000 -i ./ml/ml_play_model.py . --difficulty NORMAL --level 5
# python -m mlgame -f 480 --one-shot -i ./ml/ml_play_model.py . --difficulty NORMAL --level 5