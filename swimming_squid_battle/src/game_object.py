import math
import random

import pydantic
import pygame.sprite
from pydantic import BaseModel, field_validator
from enum import Enum
from mlgame.view.view_model import create_image_view_data, create_text_view_data
from .env import *
from .foods import Food


class LevelParams(pydantic.BaseModel):
    playground_size_w: int = 300
    playground_size_h: int = 300
    score_to_pass: int = 10
    time_to_play: int = 300

    food_1: int = 3
    food_1_max: int = 3
    food_2: int = 0
    food_2_max: int = 0
    food_3: int = 0
    food_3_max: int = 0
    garbage_1: int = 0
    garbage_1_max: int = 0
    garbage_2: int = 0
    garbage_2_max: int = 0
    garbage_3: int = 0
    garbage_3_max: int = 0
    # 補充給玩家需要的資訊
    left: int = -1
    right: int = -1
    top: int = -1
    bottom: int = -1

    @field_validator('playground_size_w', mode="before")
    def validate_playground_size_w(cls, value):
        min_size = 100
        max_size = 1200
        if value < min_size:
            return min_size
        if value > max_size:
            return max_size
        return value

    @field_validator('playground_size_h', mode="before")
    def validate_playground_size_h(cls, value):
        min_size = 100
        max_size = 650
        if value < min_size:
            return min_size
        if value > max_size:
            return max_size
        return value


# level_thresholds = [10, 15, 20, 25, 30]


class SquidState(Enum):
    NORMAL = 0
    PARALYSIS = 1
    INVINCIBLE = 2
class Motion(str,Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    NONE = "NONE"




class Squid(pygame.sprite.Sprite):
    ANGLE_TO_RIGHT = math.radians(-10)
    ANGLE_TO_LEFT = math.radians(10)
    MOTION_METHOD = {
        Motion.UP: "move_up",
        Motion.DOWN: "move_down",
        Motion.LEFT: "move_left",
        Motion.RIGHT: "move_right",
        Motion.NONE: "move_none"
    }
    def __init__(self, ai_id, x, y):
        pygame.sprite.Sprite.__init__(self)

        self._ai_num = ai_id
        self._img_id = f"squid{self._ai_num}_1"
        # self._temp_img_id = f"squid{self._ai_num}_"
        self._state = SquidState.NORMAL
        self.origin_image = pygame.Surface([SQUID_W, SQUID_H])
        self.image = self.origin_image
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self._score = 0
        self._vel = LEVEL_PROPERTIES[1]['vel']
        self._lv = 1
        self.rank = 1
        self.angle = 0
        self._last_collision = 0
        self._collision_dir = None
        self._motion = None
        self._animation_num = 1
        self._animation_direction = 1
        self._wave_degree = 0
        self.motion_method = {
            Motion.UP: self.move_up,
            Motion.DOWN: self.move_down,
            Motion.LEFT: self.move_left,
            Motion.RIGHT: self.move_right,
            Motion.NONE: self.move_none
        }
    def update(self, frame, motion:Motion):
        # for motion in motions:
        self._motion = Motion(motion)

        if self._state == SquidState.PARALYSIS:
            self._update_paralysis(frame)
        elif self._state == SquidState.INVINCIBLE:
            self._update_invincible(frame,self._motion)
        else:
            # normal action
            self._img_id = f"squid{self._ai_num}_{self._animation_num}"
            if frame % 4 == 0:
                self._animation_num += self._animation_direction
                if self._animation_num == 5 or self._animation_num == 1:
                    self._animation_direction = -self._animation_direction
            self.motion_method[motion]()
            

    def _update_paralysis(self, frame):
        if frame - self._last_collision < PARALYSIS_TIME:
            self._img_id = f"squid{self._ai_num}_hurt_{frame//4%2+1}"
            # 反彈
            # self.move(self._collision_dir)
            self.motion_method[self._collision_dir]()
        else:
            self._state = SquidState.INVINCIBLE
            self._last_collision = frame
            
    def _update_invincible(self, frame,motion):
        
        self.motion_method[motion]()
        if frame - self._last_collision < INVINCIBLE_TIME:
            self._img_id = f"squid{self._ai_num}_hurt_{frame//4%2+1}"
        else:
            self._state = SquidState.NORMAL
            self._last_collision = frame

        pass
    def move_up(self):
        """Move the squid up."""
        self.rect.centery -= self._vel
        self.angle = 0

    def move_down(self):
        """Move the squid down."""
        self.rect.centery += self._vel
        self.angle = 0

    def move_left(self):
        """Move the squid left."""
        self.rect.centerx -= self._vel
        self.angle = self.ANGLE_TO_LEFT

    def move_right(self):
        """Move the squid right."""
        self.rect.centerx += self._vel
        self.angle = self.ANGLE_TO_RIGHT
    def move_none(self):
        """Move the squid none."""
        self._wave_degree += 0.15
        self.rect.y += math.sin(self._wave_degree)
        self.angle = 0
    def move(self, motion: Motion):
        """Move the squid based on the motion command."""
        pass

    @property
    def game_object_data(self):
        return create_image_view_data(
            self._img_id,
            self.rect.x,
            self.rect.y,
            self.rect.width,
            self.rect.height,
            self.angle

        )

    def eat_food_and_change_level_and_play_sound(self, food: Food, sounds: list):
        self._score += food.score
        new_lv = get_current_level(self._score)

        if new_lv > self._lv:
            sounds.append(LV_UP_OBJ)
        elif new_lv < self._lv:
            sounds.append(LV_DOWN_OBJ)
        if new_lv != self._lv:
            self.rect.width = SQUID_W * LEVEL_PROPERTIES[new_lv]['size_ratio']
            self.rect.height = SQUID_H * LEVEL_PROPERTIES[new_lv]['size_ratio']
            self._vel = LEVEL_PROPERTIES[new_lv]['vel']
            self._lv = new_lv

    def collision_between_squids(self, collision_score, frame, sounds: list):
        if self._state == SquidState.INVINCIBLE:
            return
        self._score += collision_score
        self._last_collision = frame
        sounds.append(COLLISION_OBJ)
        self._collision_dir = random.choice([Motion.UP, Motion.DOWN, Motion.RIGHT, Motion.LEFT])

        if collision_score < 0:
            self._state = SquidState.PARALYSIS

        new_lv = get_current_level(self._score)

        if new_lv > self._lv:
            sounds.append(LV_UP_OBJ)

        elif new_lv < self._lv:
            sounds.append(LV_DOWN_OBJ)
        if new_lv != self._lv:
            self.rect.width = SQUID_W * LEVEL_PROPERTIES[new_lv]['size_ratio']
            self.rect.height = SQUID_H * LEVEL_PROPERTIES[new_lv]['size_ratio']
            self._vel = LEVEL_PROPERTIES[new_lv]['vel']
            self._lv = new_lv

    @property
    def score(self):
        return self._score

    @property
    def vel(self):
        return self._vel

    @property
    def lv(self):
        return self._lv
    @property
    def is_paralysis(self)->bool:
        return self._state==SquidState.PARALYSIS


def get_current_level(score: int) -> int:
    """
    Determines the current level based on the player's score.

    :param score: int - The current score of the player.
    :return: int - The current level of the player.
    """

    for level, threshold in enumerate(LEVEL_THRESHOLDS, start=1):
        if score < threshold:
            return min(level, 6)
    return len(LEVEL_THRESHOLDS)  # Return the next level if score is beyond all thresholds


class ScoreText(pygame.sprite.Sprite):
    def __init__(self, text, color, x, y, groups):
        pygame.sprite.Sprite.__init__(self, groups)
        self.rect = pygame.Rect(x, y, SQUID_W, SQUID_H)
        self.rect.center = (x, y)
        self._text = text
        self._color = color
        self._live_frame = 15

    def update(self):
        self._live_frame -= 1
        self.rect.centery -= 3
        if self._live_frame <= 0:
            self.kill()

    @property
    def game_object_data(self):
        return create_text_view_data(
            self._text, self.rect.centerx, self.rect.centery, self._color,
            "24px Arial BOLD")

class ForegroundText(pygame.sprite.Sprite):
    def __init__(self, text, color, x, y, groups):
        pygame.sprite.Sprite.__init__(self, groups)
        self.rect = pygame.Rect(x, y, SQUID_W, SQUID_H)
        self.rect.center = (x, y)
        self._text = text
        self._color = color
        self._live_frame = 30

    def update(self):
        self._live_frame -= 1
        self.rect.centery -= 2
        if self._live_frame <= 0:
            self.kill()

    @property
    def game_object_data(self):
        return create_text_view_data(
            self._text, self.rect.centerx, self.rect.centery, self._color,
            "32px burnfont")


class CryingStar(pygame.sprite.Sprite):
    def __init__(self, x, y, groups):
        pygame.sprite.Sprite.__init__(self, groups)
        self.rect = pygame.Rect(x, y, SQUID_H, SQUID_H)
        self.rect.center = (x, y)

        self._live_frame = 15

    def update(self):
        self._live_frame -= 1
        self.rect.centery -= 5

        if self._live_frame <= 0:
            self.kill()

    @property
    def game_object_data(self):
        return create_image_view_data(
            "star",
            self.rect.x,
            self.rect.y,
            self.rect.width,
            self.rect.height,
            angle=0.1

        )


class WindowConfig(BaseModel):
    left: int
    right: int
    top: int
    bottom: int