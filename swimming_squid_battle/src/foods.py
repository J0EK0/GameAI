import random
from math import exp, sin

import pygame.sprite
from pygame import Rect

from mlgame.view.view_model import create_image_view_data
from .env import *

FOOD1_VEL = 1
FOOD2_VEL = 2
FOOD3_VEL = 4

def bounding(value, lower_bound, upper_bound):
    return max(min(value,upper_bound),lower_bound)
def sigmoid(x):
    value = 1 / (1 + 2 * exp(-x/10))
    return value

class Food(pygame.sprite.Sprite):
    def __init__(self, group, type: FoodTypeEnum, image_id: str, image_size=None, score: int = 1):
        pygame.sprite.Sprite.__init__(self, group)
        if image_size is None:
            image_size = [FOOD_LV1_SIZE, FOOD_LV1_SIZE]
        self.image = pygame.Surface(image_size)
        self.type = type
        self.score = score
        self.rect = self.image.get_rect()
        self.angle = 0
        self.rect_float_x = 0
        self.rect_float_y = 0
        self.image_id = image_id
        self._vel=1
        self.target = None
        self._vec_x = 0
        self._vec_y = 0
        
    def update(self,playground:Rect, *args, **kwargs) -> None:
        if not(self.target):
            self.target = (
                random.randint(playground.left+10, playground.right-20),
                bounding(
                    self.rect_float_y + random.choice([1,-1])*random.randint(30, 50),
                    playground.top,
                    playground.bottom
                )
            )
            self._update_unit_vector()
        elif abs(self.rect_float_x - self.target[0]) < 8 and abs(self.rect_float_y - self.target[1]) < 8:
            self.target = (
                random.randint(playground.left+10, playground.right-20),
                bounding(
                    self.target[1]+ random.choice([1,-1])*random.randint(30, 50),
                    playground.top,
                    playground.bottom
                    )
                )
            self._update_unit_vector()


        self.rect_float_x += self._vec_x * self._vel * sigmoid(abs(self.rect_float_x - self.target[0]))
        self.rect_float_y += self._vec_y * self._vel * sigmoid(abs(self.rect_float_y - self.target[1]))
        self.rect.centerx = self.rect_float_x
        self.rect.centery = self.rect_float_y
        
        

    @property
    def game_object_data(self):
        return create_image_view_data(
            self.image_id,
            self.rect.x,
            self.rect.y,
            self.rect.width,
            self.rect.height,
        )

    def set_center_x_and_y(self, x: int, y: int):
        self.rect.centerx = x
        self.rect.centery = y
        self.rect_float_x = self.rect.centerx
        self.rect_float_y = self.rect.centery

    def _update_unit_vector(self):
        dx = self.target[0] - self.rect_float_x
        dy = self.target[1] - self.rect_float_y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        self._vec_x= dx / distance
        self._vec_y = dy / distance
        pass


class Food1(Food):
    def __init__(self, group):
        super().__init__(group, FoodTypeEnum.FOOD_1, IMG_ID_FOOD01_L, [FOOD_LV1_SIZE, FOOD_LV1_SIZE],
                         1)
        self._vel = FOOD1_VEL
        self.image_id = IMG_ID_FOOD01_R
    def update(self, playground: Rect):
        super().update(playground)
        if self._vec_x>0:
            self.image_id = IMG_ID_FOOD01_R
        elif self._vec_x<0:
            self.image_id = IMG_ID_FOOD01_L


class Food2(Food):
    def __init__(self, group):
        super().__init__(group, FoodTypeEnum.FOOD_2, IMG_ID_FOOD02_L, [FOOD_LV2_SIZE, FOOD_LV2_SIZE], 2)
        self._vel = FOOD2_VEL
        self.image_id = IMG_ID_FOOD02_R
    def update(self, playground: Rect):
        super().update(playground)
        if self._vec_x>0:
            self.image_id = IMG_ID_FOOD02_R
        elif self._vec_x<0:
            self.image_id = IMG_ID_FOOD02_L


class Food3(Food):
    def __init__(self, group):
        super().__init__(group, FoodTypeEnum.FOOD_3, IMG_ID_FOOD03_L, [FOOD_LV3_SIZE, FOOD_LV3_SIZE], 4)
        self._vel = FOOD3_VEL
        self.image_id = IMG_ID_FOOD03_R

    def update(self, playground: Rect):
        super().update(playground)
        if self._vec_x>0:
            self.image_id = IMG_ID_FOOD03_R
        elif self._vec_x<0:
            self.image_id = IMG_ID_FOOD03_L


class Garbage(Food):
    def __init__(self, group, type: FoodTypeEnum, image_id: str, image_size: list, score):
        super().__init__(group, type, image_id, image_size, score)
        self._vel = FOOD1_VEL
        self._bias_x_list = [-0.5, -0.7, -1, -1.3, 0, 1, 1.3, 0.3, 0.5, 0.7]
        self._x_degree = 0
        self._basic_vel_x = random.choice(self._bias_x_list)

    def update(self, playground: Rect):
        self._x_degree += 0.12
        self.rect_float_x += 0.5*sin(self._x_degree)+self._basic_vel_x
        self.rect_float_y += self._vel
        self.rect.centerx = self.rect_float_x
        self.rect.centery = self.rect_float_y
        self._move_to_new_position(playground)
    def _move_to_new_position(self, playground):
        if self.rect.top > playground.bottom:
            self.rect.bottom = playground.top
            self.rect_float_y = self.rect.centery
            self.rect_float_x = random.randint(playground.left, playground.right)
            self.rect.centerx = self.rect_float_x
            self._basic_vel_x = random.choice(self._bias_x_list)

        pass


class Garbage2(Garbage):
    def __init__(self, group):
        super().__init__(group, FoodTypeEnum.GARBAGE_2, "garbage02",
                         [FOOD_LV2_SIZE, FOOD_LV2_SIZE], -4)
        self._vel = FOOD2_VEL
        self._bias_x_list = [-0.5, -0.7, -1, -1.3, 0, 1, 1.3, 0.3, 0.5, 0.7]



class Garbage1(Garbage):
    def __init__(self, group):
        super().__init__(group, FoodTypeEnum.GARBAGE_1, "garbage01",
                         [FOOD_LV1_SIZE, FOOD_LV1_SIZE], -1)
        self._vel = FOOD1_VEL
        self._bias_x_list = [-0.3, -0.5, -0.7, 0, 0.3, 0.5, 0.7]


class Garbage3(Garbage):
    def __init__(self, group):
        super().__init__(group, FoodTypeEnum.GARBAGE_3, "garbage03",
                         [FOOD_LV3_SIZE, FOOD_LV3_SIZE], -10)
        self._vel = FOOD3_VEL
        self._bias_x_list = [-2.7, -1, -1.3, -1.7, 0, 1.7, 1, 1.3, 0.3, 2.7]
