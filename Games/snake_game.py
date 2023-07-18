import pygame
import random
import os
import numpy as np
import torch

from enum import Enum
from collections import namedtuple
pygame.init()

font = pygame.font.Font(os.path.join('Resources','arial.ttf'), 25)
#os.environ['SDL_AUDIODRIVER'] = 'dsp'


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
 
Point = namedtuple('Point', 'x , y')

BLOCK_SIZE=20
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

class SnakeGame:
    def __init__(self,w=640,h=480, fps=15, show_gui=True) -> None:
        self.w=w
        self.h=h
        self.fps = fps
        self.show_gui = show_gui

        # init display
        if self.show_gui:
            self.display = pygame.display.set_mode((self.w,self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()

        # Initialise the game state
        self.reset()
        
    def reset(self) -> None:
        #init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2,self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE,self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE),self.head.y)]
        self.score = 0
        self.food = None
        self._place__food()

    def _place__food(self) -> None:
        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)
        if self.food in self.snake:
            self._place__food()

    def get_input(self, action) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

    def play_step(self, action=None) -> tuple:
        # +1 for food, -1 for collision, else 0
        reward = 0

        # 1. Collect the user input
        self.get_input(action)

        # 2. Move
        self._move(self.direction)
        self.snake.insert(0,self.head)

        # 3. Check if game Over
        game_over = False 
        if self._is_collision():
            reward = -10
            game_over=True
            return game_over,self.score, reward
        # 4. Place new Food or just move
        if self.head == self.food:
            self.score+=1
            reward = 10
            self._place__food()
        else:
            self.snake.pop()
        # 5. Update UI and clock
        if self.show_gui:
            self._update_ui()
            self.clock.tick(self.fps)
        # 6. Return game Over and Display Score
        
        return game_over, self.score, reward

    def _update_ui(self) -> None:
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display,BLUE1,pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
            pygame.draw.rect(self.display,BLUE2,pygame.Rect(pt.x+4,pt.y+4,12,12))
        pygame.draw.rect(self.display,RED,pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))
        text = font.render(f"Score: {self.score}",True,WHITE)
        self.display.blit(text,[0,0])
        pygame.display.flip()

    def _move(self,direction) -> None:
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x+=BLOCK_SIZE
        elif direction == Direction.LEFT:
            x-=BLOCK_SIZE
        elif direction == Direction.DOWN:
            y+=BLOCK_SIZE
        elif direction == Direction.UP:
            y-=BLOCK_SIZE
        self.head = Point(x,y)

    def _is_collision(self,pt=None):
        if(pt is None):
            pt = self.head
        #hit boundary
        if(pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h - BLOCK_SIZE or pt.y<0):
            return True
        if(pt in self.snake[1:]):
            return True
        return False
    
    def run(self) -> None:
        game_over = False
        while not game_over:
            game_over, score, _ = self.play_step()

        print('Final Score',score)
        pygame.quit()

class SnakeGameAI(SnakeGame):
    def __init__(self, w=640, h=480, fps=20, show_gui=False) -> None:
        super().__init__(w=w, h=h, fps=fps, show_gui=show_gui)

    def get_input(self, action) -> None:
        # Action
        # [1,0,0] -> Straight
        # [0,1,0] -> Right Turn 
        # [0,0,1] -> Left Turn

        clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left Turn
        self.direction = new_dir
    
    def get_state(self) -> np.ndarray:
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_r)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            # Danger Left
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            self.food.x < self.head.x, # food is to the left
            self.food.x > self.head.x, # food is to the right
            self.food.y < self.head.y, # food is up
            self.food.y > self.head.y # food is down
        ]
            
        return np.array(state, dtype=np.float64)

if __name__=="__main__":
    game = SnakeGame()
    game.run()