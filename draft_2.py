from snake_game import *

game = SnakeGame(250)
game.fps = 5

txt_file = open("dataset.txt","w+")

for i in range(10000):
    txt_file.write(str(1))
    game.run_game()



