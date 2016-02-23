import numpy as np
import cv2

class Simulator:
    def __init__(self,reward,screen_size=64,state_space=10):
        self.number_of_actions = 3
        self.screen_size=screen_size
        self.state_space=state_space
        self.reset(reward)

    def do_action(self,action,side="right",simple_AI=False):
        if side == "right":
            #action is an int
            if action == 0:
                #do nothing
                pass
            if action == 1:
                #go up
                self.paddle_location[0] -= int(self.screen_size/(self.state_space))
            if action == 2:
                #go down
                self.paddle_location[0] += int(self.screen_size/(self.state_space))
            self.render()
            did_score = self.did_score()
            return self.screen,self.return_score(),self.points_made,did_score
        elif side == "left" and simple_AI:
            #see if the ball is above the center of paddle, move the paddle up
            if self.ball_location[0] > self.other_paddle_location[0]:
                self.other_paddle_location[0] += int(self.screen_size/(self.state_space))
            if self.ball_location[0] < self.other_paddle_location[0]:
                self.other_paddle_location[0] -= int(self.screen_size/(self.state_space))
        elif side == "left":
            #action is an int
            if action == 0:
                #do nothing
                pass
            if action == 1:
                #go up
                self.other_paddle_location[0] -= int(self.screen_size/(self.state_space))
            if action == 2:
                #go down
                self.other_paddle_location[0] += int(self.screen_size/(self.state_space))

        return None,None,None,0

    def render(self):
        #blank the screen
        self.screen = np.random.randn(self.screen.shape[0],self.screen.shape[1],self.screen.shape[2])/1000
        #draw the paddles
        self.screen[int(self.paddle_location[0]-self.paddle_size/2):int(self.paddle_location[0]+self.paddle_size/2),int(self.paddle_location[1]-1):int(self.paddle_location[1]+1),0] = 1
        self.screen[int(self.other_paddle_location[0]-self.paddle_size/2):int(self.other_paddle_location[0]+self.paddle_size/2),int(self.other_paddle_location[1]-1):int(self.other_paddle_location[1]+1),0] = 1
        #move the ball
        self.ball_location += self.ball_direction*2
        self.check_max()
        #add a reward for being on the same row as the ball
        #this is lessened by the distance of the ball from the paddle
        # self.score -= np.abs((self.ball_location[0]-self.paddle_location[0])/self.screen_size)*np.power(self.ball_location[1]/self.screen_size,2)
        #draw the ball
        xx, yy = np.mgrid[:self.screen.shape[0], :self.screen.shape[1]]
        circle = (xx - self.ball_location[0]) ** 2 + (yy - self.ball_location[1]) ** 2
        self.screen[:,:,0] += (circle < self.ball_size**2)*2 #make the ball a little circle

        self.screen = self.screen - np.mean(np.mean(self.screen,axis=0),axis=0)
        self.screen = self.screen / np.sqrt(np.var(self.screen))


    def return_score(self):
        return self.score

    def reset(self,reward=None,score=0,points_made=0):
        self.screen = np.random.randn(self.screen_size,self.screen_size,1)/100
        if reward != None:
            self.reward = reward
        self.score = score
        self.points_made = points_made
        self.paddle_size = int(self.screen_size/self.state_space)
        self.ball_size = int(self.paddle_size/4)
        # self.paddle_location = np.random.randint(self.screen_size-self.paddle_size,size=(2))
        self.paddle_location = np.zeros((2))
        self.paddle_location[0] += self.state_space
        self.paddle_location[1] = int(self.screen_size *.9)
        self.other_paddle_location = np.random.randint(self.screen_size-self.paddle_size,size=(2))
        self.other_paddle_location[1] = int(self.screen_size *.1)
        self.ball_location = np.random.uniform(size=(2))*(self.screen_size)
        self.ball_location[1] = self.screen_size / 2
        self.ball_direction = np.random.uniform(size=2)*2.0
        self.ball_direction -= 1
        if np.abs(self.ball_direction[0]) < .3:
            self.ball_direction[0] *= .3/np.abs(self.ball_direction[0])
        if np.abs(self.ball_direction[1]) < .3:
            self.ball_direction[1] *= .3/np.abs(self.ball_direction[1])
        self.render()

    def did_score(self):
        #check if the ball has gone past the paddles
        if self.ball_location[1] > self.paddle_location[1] + 1:
            self.reset(self.reward,self.score-1,points_made = self.points_made + 1) #maintain a count of how many times someone scored
            return self.score
        if self.ball_location[1] < self.other_paddle_location[1] - 1 :
            self.reset(self.reward,self.score+1,points_made = self.points_made + 1)
            return self.score
        return 0



    def check_max(self):
        #check if the ball has gone over the top or bottom, then reflect it
        if self.ball_location[0] < self.ball_size/2:
            self.ball_location[0] = np.abs(self.ball_location[0])
            self.ball_direction[0] *= -1
        if self.ball_location[0] > self.screen_size-(self.ball_size/2):
            self.ball_location[0] = self.screen_size - (self.ball_location[0] - self.screen_size)
            self.ball_direction[0] *= -1

        #check if the paddle has gone over the edge, and hold it onscreen
        if self.paddle_location[0] < self.paddle_size*.99:
            self.paddle_location[0] = int(self.paddle_size)
        if self.paddle_location[0] > self.screen_size - self.paddle_size*.99:
            self.paddle_location[0] = int(self.screen_size - self.paddle_size)
        #check if the other paddle has gone over the edge, and hold it onscreen
        if self.other_paddle_location[0] < self.paddle_size*.99:
            self.other_paddle_location[0] = int(self.paddle_size)
        if self.other_paddle_location[0] > self.screen_size - self.paddle_size*.99:
            self.other_paddle_location[0] = int(self.screen_size - self.paddle_size)

        #check if the paddle hits the ball
        if (self.ball_location[0] > self.paddle_location[0]-self.paddle_size*.9) and (self.ball_location[0] < self.paddle_location[0] + self.paddle_size*.9) and (self.ball_location[1] > self.paddle_location[1] - self.ball_size) and (self.ball_location[1] < self.paddle_location[1] + self.ball_size):
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
        if (self.ball_location[0] > self.other_paddle_location[0]-self.paddle_size*.9) and (self.ball_location[0] < self.other_paddle_location[0] + self.paddle_size*.9) and (self.ball_location[1] > self.other_paddle_location[1] - 2) and (self.ball_location[1] < self.other_paddle_location[1] + 2):
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
        if np.linalg.norm(np.abs(self.ball_direction)) < 1:
            self.ball_direction[1] = (1.245  - np.abs(self.ball_direction[0]))*(self.ball_direction[1]/np.abs(self.ball_direction[1]))

    def ball_side(self):
        if self.ball_location[1] <= self.screen_size /2:
            return "left"
        else:
            return "right"

if __name__ == "__main__":
    sim = Simulator(1)
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
