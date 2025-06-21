import sys
from os import path
sys.path.append(path.dirname(__file__))

from src.game import SwimmingSquidBattle

GAME_SETUP = {
    "game": SwimmingSquidBattle,
    # "dynamic_ml_clients":True
}
