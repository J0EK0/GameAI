import pygame
from games.swimming_squid_battle.src.foods import Food1, Food2
from games.swimming_squid_battle.src.game_object import Squid
from mlgame.game.generic import quit_or_esc
from mlgame.view.view import PygameView
from ..src.game import SwimmingSquidBattle
from ..src.game_state import TransitionState


class MockSwimmingSquidBattle(SwimmingSquidBattle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_count = 0
        self.squid1 = Squid("1P", 100, 100)
        self.squid2 = Squid("2P", 100, 100)
        self.mock_group = pygame.sprite.Group()
        
        self.squid1.eat_food_and_change_level_and_play_sound(Food2(self.mock_group),[])
        self.squid2.eat_food_and_change_level_and_play_sound(Food1(self.mock_group),[])
        
        self.current_state = TransitionState(self)
    
        
def test_transition():
    game = MockSwimmingSquidBattle()
    scene_init_info_dict = game.get_scene_init_data()
    game_view = PygameView(scene_init_info_dict)
    frame_count = 0
    while game.is_running and not quit_or_esc():
        pygame.time.Clock().tick_busy_loop(30)
        commands = game.get_keyboard_command()
        game.update(commands)
        game_progress_data = game.get_scene_progress_data()
        game_view.draw(game_progress_data)
        frame_count += 1
        # print(frame_count)
