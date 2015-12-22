import numpy
import cv2

class Simulator:
    def __init__(self,image_size,penalty):
        self.number_of_actions = 2
        self.reset(image_size,penalty)

    def do_action(self,action):
        #action is an int
        if action == 0:
            #go left
            self.bot_location[0] -= int(self.image_size/6)
        if action == 1:
            #go right
            self.bot_location[0] += int(self.image_size/6)

        self.check_max()
        self.render()
        if self.did_die():
            return self.screen,(self.score-self.penalty),True
        return self.screen,self.return_score(),None

    def render(self):
        self.score += 1
        #blank the screen
        self.screen = numpy.random.randn(self.screen.shape[0],self.screen.shape[1],self.screen.shape[2])/10
        #draw the bot
        xx, yy = numpy.mgrid[:self.screen.shape[0], :self.screen.shape[1]]
        circle = (xx - self.bot_location[1]) ** 2 + (yy - self.bot_location[0]) ** 2
        self.screen[:,:,0] += (circle < self.bot_size**2)*2 #make the bot a little circle
        #draw move the bullet down the screen
        self.bullet_location[1] += int(self.image_size/10)
        circle = (xx - self.bullet_location[1]) ** 2 + (yy - self.bullet_location[0]) ** 2
        self.screen[:,:,2] += (circle < self.bullet_size**2 )*2 #make the bullet a little circle
        self.screen = self.screen - numpy.mean(numpy.mean(self.screen,axis=0),axis=0)
        self.screen = self.screen / numpy.sqrt(numpy.var(self.screen))


    def return_score(self):
        #return the 2 norm
        return self.score # 1 - (numpy.linalg.norm(self.bot_location-self.bullet_location)/(numpy.sqrt(self.image_size**2 *2)))

    def reset(self,image_size,penalty):
        self.image_size = image_size
        self.screen = numpy.random.randn(image_size,image_size,3)/10
        self.penalty = penalty
        self.score = 0
        self.bot_size = int(image_size / 10)
        self.bullet_size = int(image_size/20)
        self.bot_location = numpy.random.randint(image_size-self.bot_size,size=(2))
        self.bot_location[1] = int(2*image_size/3)
        self.bullet_location= numpy.random.randint(image_size-self.bullet_size,size=(2))
        self.bullet_location[1] = 0
        self.check_max()
        self.render()

    def did_die(self):
        #first check if the bot and bullet are overlapping
        blue_plus_red = self.screen[:,:,0] + self.screen[:,:,2]
        if numpy.max(blue_plus_red) > 12:
            #there's an overlap!
            return True
        return False

    def check_max(self):
        #check if the bot has gone over the edge
        self.bot_location[0] = numpy.min([self.bot_location[0],self.image_size - self.bot_size])
        self.bot_location[0] = numpy.max([self.bot_size,self.bot_location[0]])

        self.bot_location[1] = numpy.min([self.bot_location[1],self.image_size - self.bot_size])
        self.bot_location[1] = numpy.max([self.bot_size,self.bot_location[1]])
        #if the bullet has exceeded the bottom
        self.bullet_location[0] = numpy.min([self.bullet_location[0],self.image_size - self.bullet_size])
        self.bullet_location[0] = numpy.max([self.bullet_size,self.bullet_location[0]])

        #self.bullet_location[1] = numpy.min([self.bullet_location[1],self.image_size - self.bullet_size])
        if (self.image_size - self.bullet_size) < self.bullet_location[1]:
            #make a new bullet
            self.bullet_location[0] = numpy.random.randint(self.image_size-self.bullet_size)
            self.bullet_location[1] = 0
        self.bullet_location[1] = numpy.max([self.bullet_size/2,self.bullet_location[1]])


if __name__ == "__main__":
    sim = Simulator(256,10)
    cv2.imshow('sim',sim.screen)
    cv2.waitKey(1000)
    screen,score,end = sim.do_action(1)
    screen,score,end = sim.do_action(1)
    screen,score,end = sim.do_action(1)
    for i in range(10):
        action = numpy.random.randint(2)
        screen,score,end = sim.do_action(action)
        cv2.imshow('sim',screen)
        print(score)
        cv2.waitKey(1000)
        screen,score,end = sim.do_action(action)
        cv2.imshow('sim',screen)
        print(score)
        cv2.waitKey(1000)
        screen,score,end = sim.do_action(action)
        cv2.imshow('sim',screen)
        print(score)
        cv2.waitKey(1000)
