import numpy
import cv2

class Simulator:
    def __init__(self,image_size,reward):
        self.reset(image_size,reward)

    def do_action(self,action):
        #action is an int
        if action == 0:
            #go left
            self.bot_location[1] -= 4
        if action == 1:
            #go right
            self.bot_location[1] += 4
        if action == 2:
            #go up
            self.bot_location[0] -= 4
        if action == 3:
            #go down
            self.bot_location[0] += 4

        self.check_max()
        self.render()
        if self.did_win():
            return self.screen,self.reward
        return self.screen,self.score()

    def render(self):
        #blank the screen
        self.screen = numpy.random.randn(self.screen.shape[0],self.screen.shape[1],self.screen.shape[2])/10
        #draw the bot
        xx, yy = numpy.mgrid[:self.screen.shape[0], :self.screen.shape[1]]
        circle = (xx - self.bot_location[1]) ** 2 + (yy - self.bot_location[0]) ** 2
        self.screen[:,:,0] += (circle < self.bot_size**2) #numpy.logical_and(circle < (self.bot_size**2 ), 1)
        #self.screen[self.bot_location[0],self.bot_location[1]] = 1
        circle = (xx - self.goal_location[1]) ** 2 + (yy - self.goal_location[0]) ** 2
        self.screen[:,:,2] += (circle < self.goal_size**2 ) #numpy.logical_and(circle < (self.goal_size**2 ), 1)

    def score(self):
        #return the 2 norm
        return 1 - (numpy.linalg.norm(self.bot_location-self.goal_location)/(numpy.sqrt(self.image_size**2 *2)))

    def reset(self,image_size,reward):
        self.image_size = image_size
        self.screen = numpy.random.randn(image_size,image_size,3)/10
        self.reward = reward
        self.bot_size = int(image_size / 10)
        self.goal_size = int(image_size/20)
        self.bot_location = numpy.random.randint(image_size-self.bot_size,size=(2))
        self.goal_location= numpy.random.randint(image_size-self.goal_size,size=(2))
        self.check_max()
        self.render()

    def did_win(self):
        #first check if the bot and goal are overlapping
        blue_plus_red = self.screen[:,:,0] + self.screen[:,:,2]
        if numpy.max(blue_plus_red) > 1.8:
            #there's an overlap!
            return True
        return False

    def check_max(self):
        #check if the bot has gone over the edge
        if self.bot_location[0] > (self.image_size - self.bot_size):
            #its too far down, reset it
            self.bot_location[0] = (self.image_size - self.bot_size)

        if self.bot_location[0] < self.bot_size:
            #its too far up, reset it
            self.bot_location[0] = self.bot_size

        if self.bot_location[1] > (self.image_size - self.bot_size):
            #its too far right, reset it
            self.bot_location[1] = (self.image_size - self.bot_size)

        if self.bot_location[1] < self.bot_size:
            #its too far left, reset it
            self.bot_location[1] = self.bot_size

        #check if the goal has gone over the edge
        if self.goal_location[0] > (self.image_size - self.goal_size):
            #its too far down, reset it
            self.goal_location[0] = (self.image_size - self.goal_size)

        if self.goal_location[0] < self.goal_size:
            #its too far up, reset it
            self.goal_location[0] = self.goal_size

        if self.goal_location[1] > (self.image_size - self.goal_size):
            #its too far right, reset it
            self.goal_location[1] = (self.image_size - self.goal_size)

        if self.goal_location[1] < self.goal_size:
            #its too far left, reset it
            self.goal_location[1] = self.goal_size

if __name__ == "__main__":
    sim = Simulator(256,10)
    cv2.imshow('sim',sim.screen)
    cv2.waitKey(1000)
    screen,score = sim.do_action(1)
    screen,score = sim.do_action(1)
    screen,score = sim.do_action(1)
    for i in range(10):
        action = numpy.random.randint(4)
        screen,score = sim.do_action(action)
        screen,score = sim.do_action(action)
        screen,score = sim.do_action(action)
        cv2.imshow('sim',screen)
        print(score)
        cv2.waitKey(1000)
