import os
import pygame
from mlgame.game.generic import quit_or_esc
from mlgame.view.view import PygameView
from ..src.game import SwimmingSquidBattle
from ..src.game_state import RunningState


class MockSwimmingSquidBattle(SwimmingSquidBattle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,level_file=os.path.join(os.path.dirname(__file__),"..","levels" ,"template.json"), **kwargs)
        self.frame_count = 0
        self.mock_group = pygame.sprite.Group()
        # self.squid1.eat_food_and_change_level_and_play_sound(Food2(self.mock_group),[])
        # self.squid2.eat_food_and_change_level_and_play_sound(Food1(self.mock_group),[])

        self._game_times=5
        self.set_game_state(RunningState.PLAYING)


def test_ending():
    game = MockSwimmingSquidBattle()
    scene_init_info_dict = game.get_scene_init_data()
    game_view = PygameView(scene_init_info_dict)
    frame_count = 0
    clock = pygame.time.Clock()
    while game.is_running and not quit_or_esc():
        clock.tick(30)

        commands = game.get_keyboard_command()
        game.update(commands)
        game_progress_data = game.get_scene_progress_data()
        game_view.draw(game_progress_data)
        frame_count += 1
        # print(frame_count)
