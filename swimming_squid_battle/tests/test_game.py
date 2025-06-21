from ..src.game import SwimmingSquidBattle, remap

from ..src.game_state import EndingState, OpeningState, RunningState, TransitionState
def test_set_game_state():
    game = SwimmingSquidBattle()
    game.set_game_state(RunningState.OPENING)
    assert isinstance(game.current_state,OpeningState)
    game.set_game_state(RunningState.TRANSITION)
    assert isinstance(game.current_state,TransitionState)
    game.set_game_state(RunningState.ENDING)
    assert isinstance(game.current_state,EndingState)

def test_set_game_state_invalid():
    game = SwimmingSquidBattle()
    game.set_game_state(RunningState.RESET)
    assert game.current_state is None
    game.set_game_state(RunningState.PLAYING)
    assert game.current_state is None


def test_remap():
    assert remap(100, 0, 100, 0, 1000) == 1000
    assert remap(0, 0, 100, 0, 1000) == 0
    assert remap(50, 0, 50, 0, 1000) == 1000
    assert remap(-10, 0, 50, 0, 1000) == 0


