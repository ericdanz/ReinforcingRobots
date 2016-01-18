import numpy
import cv2

class Simulator:
    def __init__(self,reward):
        self.number_of_actions = 3
        self.reset(reward)

    def do_action(self,action,side="right"):
        if side == "right":
            #action is an int
            if action == 0:
                #do nothing
                pass
            if action == 1:
                #go up
                self.paddle_location[0] -= int(self.image_size/10)
            if action == 2:
                #go down
                self.paddle_location[0] += int(self.image_size/10)
            self.render()
            did_score = self.did_score()
            if did_score != 0:
                return self.screen,self.return_score(),self.points_made,did_score
            else:
                return self.screen,self.return_score(),self.points_made,did_score
        elif side == "left":
            #action is an int
            if action == 0:
                #do nothing
                pass
            if action == 1:
                #go up
                self.other_paddle_location[0] -= int(self.image_size/10)
            if action == 2:
                #go down
                self.other_paddle_location[0] += int(self.image_size/10)

            #self.render() #these only happen when the 'right' side moves
            #self.did_score()
            return None,None,None,0 #self.screen,self.return_score()

    def render(self):
        #blank the screen
        self.screen = numpy.random.randn(self.screen.shape[0],self.screen.shape[1],self.screen.shape[2])/1000
        #draw the paddles
        self.screen[int(self.paddle_location[0]-self.paddle_size/2):int(self.paddle_location[0]+self.paddle_size/2),int(self.paddle_location[1]-1):int(self.paddle_location[1]+1),0] = 1
        self.screen[int(self.other_paddle_location[0]-self.paddle_size/2):int(self.other_paddle_location[0]+self.paddle_size/2),int(self.other_paddle_location[1]-1):int(self.other_paddle_location[1]+1),0] = 1
        #move the ball
        self.ball_location += self.ball_direction*2
        self.check_max()
        #draw the ball
        xx, yy = numpy.mgrid[:self.screen.shape[0], :self.screen.shape[1]]
        circle = (xx - self.ball_location[0]) ** 2 + (yy - self.ball_location[1]) ** 2
        self.screen[:,:,0] += (circle < self.ball_size**2)*2 #make the ball a little circle

        self.screen = self.screen - numpy.mean(numpy.mean(self.screen,axis=0),axis=0)
        self.screen = self.screen / numpy.sqrt(numpy.var(self.screen))


    def return_score(self):
        #return the 2 norm
        return self.score

    def reset(self,reward=None,score=0,points_made=0):
        self.image_size = 128
        self.screen = numpy.random.randn(self.image_size,self.image_size,1)/100
        if reward != None:
            self.reward = reward
        self.score = score
        self.points_made = points_made
        self.paddle_size = int(self.image_size/10)
        self.ball_size = int(self.paddle_size/4)
        self.paddle_location = numpy.random.randint(self.image_size-self.paddle_size,size=(2))
        self.paddle_location[1] = int(self.image_size *.9)
        self.other_paddle_location = numpy.random.randint(self.image_size-self.paddle_size,size=(2))
        self.other_paddle_location[1] = int(self.image_size *.1)
        self.ball_location = numpy.random.uniform(size=(2))*(self.image_size)
        self.ball_location[1] = self.image_size / 2
        self.ball_direction = numpy.random.uniform(size=2)*2.0
        self.ball_direction -= 1
        # print(self.ball_direction)
        if numpy.abs(self.ball_direction[0]) < .3:
            self.ball_direction[0] *= .3/numpy.abs(self.ball_direction[0])
        if numpy.abs(self.ball_direction[1]) < .3:
            self.ball_direction[1] *= .3/numpy.abs(self.ball_direction[1])
        # print(self.ball_direction)
        self.render()

    def did_score(self):
        #check if the ball has gone past the paddles
        if self.ball_location[1] > self.paddle_location[1] + 1:
            # self.score -= self.reward
            self.reset(self.reward,points_made = self.points_made + 1) #maintain a count of how many times someone scored
            return -1
        if self.ball_location[1] < self.other_paddle_location[1] - 1 :
            # self.score += self.reward
            self.reset(self.reward,points_made = self.points_made + 1)
            return 1
        return 0



    def check_max(self):
        #check if the ball has gone over the top or bottom, then reflect it
        if self.ball_location[0] < self.ball_size/2:
            self.ball_location[0] = numpy.abs(self.ball_location[0])
            self.ball_direction[0] *= -1
        if self.ball_location[0] > self.image_size-(self.ball_size/2):
            self.ball_location[0] = self.image_size - (self.ball_location[0] - self.image_size)
            self.ball_direction[0] *= -1

        #check if the paddle has gone over the edge, and hold it onscreen
        if self.paddle_location[0] < self.paddle_size*.9:
            self.paddle_location[0] = int(self.paddle_size)
        if self.paddle_location[0] > self.image_size - self.paddle_size*.9:
            self.paddle_location[0] = int(self.image_size - self.paddle_size)
        #check if the other paddle has gone over the edge, and hold it onscreen
        if self.other_paddle_location[0] < self.paddle_size*.9:
            self.other_paddle_location[0] = int(self.paddle_size)
        if self.other_paddle_location[0] > self.image_size - self.paddle_size*.9:
            self.other_paddle_location[0] = int(self.image_size - self.paddle_size)

        #check if the paddle hits the ball
        if (self.ball_location[0] > self.paddle_location[0]-self.paddle_size) and (self.ball_location[0] < self.paddle_location[0] + self.paddle_size) and (self.ball_location[1] > self.paddle_location[1] - 2) and (self.ball_location[1] < self.paddle_location[1] + 2):
            self.ball_direction[1] *= -1
            #check for edge hits, and increase the y velocity and decrease the x velocity
            if (self.ball_location[0] < self.paddle_location[0]-self.paddle_size/3):
                #increase neg y velocity
                self.ball_direction[0] -= 1
            #check for edge hits, and increase the y velocity and decrease the x velocity
            if (self.ball_location[0] > self.paddle_location[0] + self.paddle_size/3):
                #increase neg y velocity
                self.ball_direction[0] += 1
        #check if the other paddle hits the ball
        if (self.ball_location[0] > self.other_paddle_location[0]-self.paddle_size) and (self.ball_location[0] < self.other_paddle_location[0] + self.paddle_size) and (self.ball_location[1] > self.other_paddle_location[1] - 2) and (self.ball_location[1] < self.other_paddle_location[1] + 2):
            self.ball_direction[1] *= -1
            #check for edge hits, and increase the y velocity and decrease the x velocity
            if (self.ball_location[0] < self.other_paddle_location[0]-self.paddle_size/3):
                #increase neg y velocity
                self.ball_direction[0] -= 1
            #check for edge hits, and increase the y velocity and decrease the x velocity
            if (self.ball_location[0] > self.other_paddle_location[0] + self.paddle_size/3):
                #increase neg y velocity
                self.ball_direction[0] += 1
        #make sure total velocity stays the same
        if numpy.linalg.norm(numpy.abs(self.ball_direction)) < 1:
            self.ball_direction[1] = (1.245  - numpy.abs(self.ball_direction[0]))*(self.ball_direction[1]/numpy.abs(self.ball_direction[1]))

    def ball_side(self):
        if self.ball_location[1] <= self.image_size /2:
            return "left"
        else:
            return "right"

if __name__ == "__main__":
    sim = Simulator(10)
    cv2.imshow('Phong!',sim.screen)
    cv2.waitKey(1000)
    screen,score = sim.do_action(2)
    screen,score = sim.do_action(2)
    screen,score = sim.do_action(2)
    screen,score = sim.do_action(2)
    screen,score = sim.do_action(2)
    screen,score = sim.do_action(2)
    for i in range(1000):
        screen,score = sim.do_action(0)
        cv2.imshow('sim',cv2.resize(screen,(0,0),fx=4,fy=4))
        key = cv2.waitKey(80)
        print(key)
        if key == 63232:
            screen,score = sim.do_action(1)
        elif key == 63233:
            screen,score = sim.do_action(2)
        else:
            screen,score = sim.do_action(0)
        print(score)
