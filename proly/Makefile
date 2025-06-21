
train_RB:
	python -m mlgame3d --decision-period 10 -f 60 -ts 10 -i ./ml/ml_play_RB.py -i hidden -i hidden -i hidden -e 1000 -ng -gp mud_pit 2 -gp checkpoint 8 -gp items 0 -gp audio false ./Proly.app

test:
	python -m mlgame3d --decision-period 10 -f 60 -ts 10 -i ./ml/ml_play.py -i ./ml/ml_play_battle.py -i hidden -i hidden -e 100 -gp mud_pit 2 -gp checkpoint 8 -gp items 0 -gp audio false ./Proly.app

train:
	python -m mlgame3d --decision-period 10 -f 60 -ts 10 -i ./ml/ppo_mlplay_template.py -i hidden -i hidden -i hidden -e 1000 -gp mud_pit 2 -gp checkpoint 8 -gp items 0 -gp audio false ./Proly.app

plot:
	python ml/plot.py

clean:
	rm -rf models
	rm log/player1_metrics.csv
	rm log/move_vectors.csv


# conda activate proly
# python -m mlgame3d -f 60 -ts 10 -i ./ml/ml_play.py -i hidden -i hidden -i hidden -e 100 -gp checkpoint 8 -gp items 0 -gp audio false ./Proly.app
