import pygame
import pygame.surfarray as surfarray
import math
import torch
import torchvision.transforms as T
import numpy as np
from random import randint
from random import choice

pygame.init()

class SnakeGame():
    def __init__(self, screen_size):
        self.screen_size = screen_size
        self.fps = 0
        self.screen = pygame.display.set_mode((self.screen_size,self.screen_size))
        self.clock = pygame.time.Clock()
        self.running = True
        self.snake_x_pos = randint(0,(screen_size/10)-1)*10
        self.snake_y_pos = randint(0,(screen_size/10)-1)*10
        self.food_x_pos = randint(0,(screen_size/10)-1)*10
        self.food_y_pos = randint(0,(screen_size/10)-1)*10
        self.counter = 500
        self.coordinates_list = []
        self.coordinates_list.append((self.snake_x_pos,self.snake_y_pos))
        self.snake_length = 2
        self.score = 0
        self.direction = choice(["RIGHT","LEFT","UP", "DOWN"])
        ############DQN#################
        self.done = False
        self.reward = 0
        self.dist = math.hypot(self.snake_x_pos-self.food_x_pos, self.snake_y_pos-self.food_y_pos)/self.screen_size
        self.dist_next = 0

    def game_reset(self):
        self.score = 0
        self.counter = 500
        self.snake_x_pos = randint(0,(self.screen_size/10)-1)*10
        self.snake_y_pos = randint(0,(self.screen_size/10)-1)*10
        self.food_x_pos = randint(0,(self.screen_size/10)-1)*10
        self.food_y_pos = randint(0,(self.screen_size/10)-1)*10
        self.coordinates_list = []
        self.snake_length = 2
        self.coordinates_list.append((self.snake_x_pos,self.snake_y_pos))
        self.direction = choice(["RIGHT","LEFT","UP", "DOWN"])
        ############DQN#################
        self.done = False
        self.reward = 0
        self.dist = math.hypot(self.snake_x_pos-self.food_x_pos, self.snake_y_pos-self.food_y_pos)/self.screen_size
        self.dist_next = 0

    def run_game(self):
        self.reward = 0
        self.counter -= 1
        if self.counter <= 0:
            self.done = True
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                self.running = False

            elif i.type == pygame.KEYDOWN:
                if i.key == pygame.K_w:
                    self.direction = "UP"
                elif i.key == pygame.K_s:
                    self.direction = "DOWN"
                elif i.key == pygame.K_a:
                    self.direction = "LEFT"
                elif i.key == pygame.K_d:
                    self.direction = "RIGHT"

        if self.direction == "UP":
            self.snake_y_pos -= 10
        elif self.direction == "DOWN":
            self.snake_y_pos += 10
        if self.direction == "LEFT":
            self.snake_x_pos -= 10
        elif self.direction == "RIGHT":
            self.snake_x_pos += 10

        self.coordinates_list.append((self.snake_x_pos, self.snake_y_pos))
        if self.snake_x_pos == self.food_x_pos and self.snake_y_pos == self.food_y_pos:
            self.reward += 50
            self.counter = 500
            self.score += 1
            while (self.food_x_pos, self.food_y_pos) in self.coordinates_list:
                self.food_x_pos = randint(0, (self.screen_size / 10) - 1) * 10
                self.food_y_pos = randint(0, (self.screen_size / 10) - 1) * 10
            self.snake_length += 1

        if len(self.coordinates_list) >= self.snake_length:
            self.coordinates_list.pop(0)

        self.screen.fill((0,0,0))
        pygame.draw.rect(self.screen, (255, 255, 255), (self.snake_x_pos, self.snake_y_pos, 10, 10))
        pygame.draw.rect(self.screen, (125, 125, 125), (self.food_x_pos, self.food_y_pos, 10, 10))
        for i in self.coordinates_list[:-1]:
            pygame.draw.rect(self.screen, (255, 255, 255), (i[0], i[1], 10, 10))

        if self.snake_x_pos >= self.screen_size or self.snake_y_pos >= self.screen_size or self.snake_x_pos < 0 or self.snake_y_pos < 0:
            self.running = False
            self.done = True
            self.reward -= 100
        if (self.snake_x_pos, self.snake_y_pos) in self.coordinates_list[:-1]:
            self.running = False
            self.done = True
            self.reward -= 100

        self.dist = math.hypot(self.snake_x_pos-self.food_x_pos, self.snake_y_pos-self.food_y_pos)/self.screen_size
        if self.dist < self.dist_next:
            self.reward += 1
        else:
            self.reward -= 1
        self.dist_next = self.dist

        pygame.display.update()
        self.clock.tick(self.fps)

    def take_action(self,action):
        if action == 0 and self.direction != "DOWN":
            self.direction = "UP"
        if action == 1 and self.direction != "UP":
            self.direction = "DOWN"
        if action == 2 and self.direction != "RIGHT":
            self.direction = "LEFT"
        if action == 3 and self.direction != "LEFT":
            self.direction = "RIGHT"

        #DATASET
    def get_current_state(self):
        '''
        if self.snake_x_pos >= self.screen_size or self.snake_y_pos >= self.screen_size or self.snake_x_pos < 0 or self.snake_y_pos < 0:
            screen_tensor = torch.zeros(20,20)
        if (self.snake_x_pos,self.snake_y_pos) in self.coordinates_list[:-1]:
            screen_tensor = torch.zeros(20,20)
        '''
        surface_array = surfarray.array2d(self.screen)
        screen_data = np.delete(surface_array,[i for i in range(self.screen_size) if i%10 != 0],0)
        screen_data = np.delete(screen_data,[i for i in range(self.screen_size) if i%10 != 0],1)
        screen_data = np.divide(screen_data,2**24)
        screen_tensor = torch.tensor(screen_data,dtype = torch.float32)
        return screen_tensor

    def get_current_state_2(self):
        data_tensor = torch.zeros(21)
        # directions
        if self.direction == "UP":
            data_tensor[0] = 1.0
        if self.direction == "DOWN":
            data_tensor[1] = 1.0
        if self.direction == "LEFT":
            data_tensor[2] = 1.0
        if self.direction == "RIGHT":
            data_tensor[3] = 1.0

        # food location
        if self.food_x_pos < self.snake_x_pos:
            data_tensor[4] = 1
        if self.food_x_pos > self.snake_x_pos:
            data_tensor[5] = 1
        if self.food_y_pos < self.snake_y_pos:
            data_tensor[6] = 1
        if self.food_y_pos > self.snake_y_pos:
            data_tensor[7] = 1

        data_tensor[8] = math.hypot(self.snake_x_pos-self.food_x_pos, self.snake_y_pos-self.food_y_pos)/self.screen_size

        # obstacles detection
        if self.snake_x_pos - 20 < 0 or (self.snake_x_pos-20, self.snake_y_pos) in self.coordinates_list:
            data_tensor[9] = 1
        if self.snake_x_pos - 10 == 0 or (self.snake_x_pos-10, self.snake_y_pos) in self.coordinates_list:
            data_tensor[10] = 1
        if self.snake_x_pos + 20 >= self.screen_size or (self.snake_x_pos+20, self.snake_y_pos) in self.coordinates_list:
            data_tensor[11] = 1
        if self.snake_x_pos + 10 >= self.screen_size or (self.snake_x_pos+10, self.snake_y_pos) in self.coordinates_list:
            data_tensor[12] = 1

        if self.snake_y_pos-10 <= 0 or (self.snake_x_pos, self.snake_y_pos-20) in self.coordinates_list:
            data_tensor[13] = 1
        if self.snake_y_pos-10 == 0 or (self.snake_x_pos, self.snake_y_pos-10) in self.coordinates_list:
            data_tensor[14] = 1
        if self.snake_y_pos + 20 >= self.screen_size or (self.snake_x_pos, self.snake_y_pos + 20) in self.coordinates_list:
            data_tensor[15] = 1
        if self.snake_y_pos + 10 >= self.screen_size or (self.snake_x_pos, self.snake_y_pos + 10) in self.coordinates_list:
            data_tensor[16] = 1

        # obstacles line trace
        x_obstacles_list = [x for x,y in self.coordinates_list if y == self.snake_y_pos]
        y_obstacles_list = [y for x,y in self.coordinates_list if x == self.snake_x_pos]

        data_tensor[17] = len([i for i in x_obstacles_list if i < x_obstacles_list[-1]])*10/self.screen_size
        data_tensor[18] = len([i for i in x_obstacles_list if i > x_obstacles_list[-1]])*10/self.screen_size
        data_tensor[19] = len([i for i in y_obstacles_list if i < y_obstacles_list[-1]])*10/self.screen_size
        data_tensor[20] = len([i for i in y_obstacles_list if i > y_obstacles_list[-1]])*10/self.screen_size


        return data_tensor.to(torch.device("cpu"), non_blocking=True)
