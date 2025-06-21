from enum import Enum
from math import sin
from random import randint

from mlgame.view.audio_model import MusicProgressSchema

from .env import HEIGHT, IMG_ID_DOT_LOSE, IMG_ID_DOT_NONE, IMG_ID_DOT_WIN, IMG_ID_ENDING_TROPHY, IMG_ID_OPENNING_BG, IMG_ID_OPENNING_LOGO, IMG_ID_TRANSITION_BG, IMG_ID_TRANSITION_CROWN, IMG_ID_TRANSITION_P1, IMG_ID_TRANSITION_P2, PASS_OBJ, WIDTH
from mlgame.game.paia_game import GameState
from mlgame.view.view_model import create_image_view_data, create_scene_progress_data, create_text_view_data


class EndingState(GameState):
    def __init__(self,game):
        self._game = game
        self.frame_count = 0
        self._info_text = {}
        self._sound = [PASS_OBJ]
        self._p1_point = 0
        self._p2_point = 0
        self._crown_degree = 0
        self._crown_bias = sin(self._crown_degree)
        self.reset()
    def update(self):
        self._crown_degree += 0.15
        self._crown_bias = 10*sin(self._crown_degree)
        if self.frame_count == 0:
            self._sound = [PASS_OBJ]
        else:
            self._sound = []

        if 0 < self.frame_count < 120:

            pass
        elif 90 <= self.frame_count:
            self.reset()
            self._game.set_game_state(RunningState.RESET)
            
        self.frame_count += 1
    def get_scene_progress_data(self):
        object_list = []
        self._p1_point = self._game._winner.count("1P")
        self._p2_point = self._game._winner.count("2P")
        player_x = WIDTH/4
        player1_y = HEIGHT/4+70
        player2_y =HEIGHT/4+260
        dot1_y = player1_y+62
        dot2_y = player2_y+62
        for i in range(self._game._game_times):
            dot_x = player_x+200+i*70
            if i < len(self._game._winner):
                if self._game._winner[i] == "1P":
                    object_list.extend(
                        [
                            # 1P
                            create_image_view_data(IMG_ID_DOT_WIN, dot_x, dot1_y,50,50),
                            # 2P
                            create_image_view_data(IMG_ID_DOT_LOSE, dot_x, dot2_y,50,50),
                        ]
                    )
                elif self._game._winner[i] == "2P":
                    object_list.extend(
                        [
                            # 1P
                            create_image_view_data(IMG_ID_DOT_LOSE, dot_x, dot1_y,50,50),
                            # 2P
                            create_image_view_data(IMG_ID_DOT_WIN, dot_x, dot2_y,50,50),
                        ]
                    )
                else:
                    object_list.extend(
                        [
                            create_image_view_data(IMG_ID_DOT_NONE, dot_x, dot1_y,50,50),
                            create_image_view_data(IMG_ID_DOT_NONE, dot_x, dot2_y,50,50),
                        ]
                    )
                
            else:
                object_list.extend(
                        [
                            create_image_view_data(IMG_ID_DOT_NONE, dot_x, dot1_y,50,50),
                            create_image_view_data(IMG_ID_DOT_NONE, dot_x, dot2_y,50,50),
                        ]
                    )
        
        
        object_list.extend( [
            create_image_view_data(IMG_ID_TRANSITION_P1, player_x, player1_y, 164, 164),
            create_image_view_data(IMG_ID_TRANSITION_P2, player_x, player2_y, 164, 164),
        ])
        
        # crown
        if self._p1_point > self._p2_point:
            object_list.append(
                create_image_view_data(IMG_ID_TRANSITION_CROWN, player_x-60, player1_y-40+self._crown_bias, 114, 80),
            )
        elif self._p1_point < self._p2_point:
            object_list.append(
                create_image_view_data(IMG_ID_TRANSITION_CROWN, player_x-60 , player2_y-40+self._crown_bias, 114, 80),
            )
        
        
        return create_scene_progress_data(
            frame=self.frame_count,
            background=[
                create_image_view_data(IMG_ID_ENDING_TROPHY, WIDTH/2-314, HEIGHT/2-511/2, 628, 511),
                ],
            object_list=object_list,
            foreground=[],
            toggle=[],
            musics=[], sounds=self._sound
            )
    def reset(self):
        self.frame_count = 0
        self._info_text = create_text_view_data("Ending", 130, 0, "#EEEEEE", "64px Consolas BOLD")
        self._sound = [PASS_OBJ]


class TransitionState(GameState):
    def __init__(self,game):
        self._game = game
        self.frame_count = 0
        self._info_text = {}
        self._sound = [PASS_OBJ]
        self._crown_x = 0
        self._crown_y = HEIGHT/2-125
        self._crown_y_degree = 0
        self._crown_y_bias = 0
        self._p1_score = 0
        self._p2_score = 0
        self.reset()
    def update(self):
        self._p1_score = self._game.squid1.score
        self._p2_score = self._game.squid2.score
        self._crown_y_degree += 0.15
        self._crown_y_bias = 10*sin(self._crown_y_degree)
        if self._p1_score > self._p2_score:
            self._crown_x = WIDTH/2-368
        elif self._p1_score < self._p2_score:
            self._crown_x = WIDTH/2+124
        
        if self.frame_count == 0:
            self._sound = [PASS_OBJ]
        else:
            self._sound = []

        if self.frame_count < 120:
            pass
        elif 91 <= self.frame_count:
            self.reset()
            self._game.set_game_state(RunningState.PLAYING)
        self.frame_count += 1
        
    def get_scene_progress_data(self):
        return create_scene_progress_data(
            frame=self.frame_count,
            background=[
                create_image_view_data(IMG_ID_TRANSITION_BG, 0, 0, 1280, 768),
                ],
            object_list=[
                create_image_view_data(IMG_ID_TRANSITION_P1, WIDTH/2-328, HEIGHT/2-82, 164, 164),
                create_image_view_data(IMG_ID_TRANSITION_P2, WIDTH/2+164, HEIGHT/2-82, 164, 164),
                create_text_view_data(f"{self._p1_score:03d}pt", WIDTH/2-328, HEIGHT/2+164, "#EEEEEE", "48px Burnfont BOLD"),
                create_text_view_data(f"{self._p2_score:03d}pt", WIDTH/2+164, HEIGHT/2+164, "#EEEEEE", "48px Burnfont BOLD"),
                create_image_view_data(IMG_ID_TRANSITION_CROWN, self._crown_x, self._crown_y+self._crown_y_bias, 114, 80),
                ],
            foreground=[
                create_text_view_data(f"Round {len(self._game._winner)}", WIDTH/2-140, HEIGHT/2-300, "#EEEEEE", "64px Burnfont BOLD"),
                ],
            toggle=[],
            musics=[], sounds=self._sound
            )
    def reset(self):
        self.frame_count = 0
        self._info_text = create_text_view_data("Ready", 230, 0, "#EEEEEE", "64px Consolas BOLD")
        self._sound.append(PASS_OBJ)

class OpeningState(GameState):
    def __init__(self, game: 'SwimmingSquidBattle'):
        self._game = game
        self.frame_count = 0
        # self._ready_text = create_text_view_data("Ready", 300, 300, "#EEEEEE", "64px Consolas BOLD")
        # self._go_text = create_text_view_data("Go! ", -300, -360, "#EEEEEE", "64px Consolas BOLD")
        self._logo_bias_degree = 0
        self._logo_bias = sin(self._logo_bias_degree)
    def update(self):
        if self.frame_count < 90:
            self._logo_bias_degree += 0.15
            self._logo_bias = 15*sin(self._logo_bias_degree)
        elif 90 <= self.frame_count:
            self.frame_count=0
            self.reset()
            self._game.set_game_state(RunningState.PLAYING)
        self.frame_count += 1

    def get_scene_progress_data(self):
        return create_scene_progress_data(
            frame=self.frame_count,
            background=[
                create_image_view_data(IMG_ID_OPENNING_BG, 0, 0, 1280, 768),
                create_image_view_data(IMG_ID_OPENNING_LOGO, 1280/2-506/2, 768/2-256/2+self._logo_bias, 506, 256),
                ],
            object_list=[
                # self._ready_text, self._go_text
                ],
            foreground=[],
            toggle=[],
            musics=[MusicProgressSchema(music_id=f"bgm0{randint(1, 3)}").__dict__] if self.frame_count == 2 else [],
            sounds=[]
            )

    def reset(self):
        self.frame_count = 0
        self._logo_bias_degree = 0
        self._logo_bias = sin(self._logo_bias_degree)

class RunningState(Enum):
    OPENING = 0
    TRANSITION = 1
    ENDING = 2
    PLAYING = 3
    RESET = 4
