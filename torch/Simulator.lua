require 'torch'
local class = require 'class'

local Simulator = torch.class('Simulator')


function Simulator:__init(image_size,reward)
  self:reset(image_size,reward)
end

function Simulator:reset(image_size,reward)
  self.image_size = image_size
  self.screen = torch.randn(3,image_size,image_size)/10
  self.reward = reward
  self.bot_size = torch.ceil(image_size / 10)
  self.goal_size = torch.ceil(image_size/20)
  self.bot_location = torch.rand(2):mul(image_size):add(1):int() --make random location
  self.goal_location = torch.rand(2):mul(image_size):add(1):int() --from [1,imagesize]
  self.upper_boundary = torch.zeros(2):add(image_size):add(-self.bot_size):int() --use bot size since it is larger
  self.lower_boundary = torch.zeros(2):add(self.bot_size):int()
  self:check_max()
  self:render()
end

function Simulator:check_max()
  --check if the bot has gone over the edge
  self.bot_location = torch.cmin(self.bot_location,self.upper_boundary)
  self.bot_location = torch.cmax(self.bot_location,self.lower_boundary)
  --check if the goal has gone over the edge
  self.goal_location = torch.cmin(self.goal_location,self.upper_boundary)
  self.goal_location = torch.cmax(self.goal_location,self.lower_boundary)
end

function Simulator:render()
  --blank the screen
  self.screen = torch.randn(3,self.image_size,self.image_size)/10
  --draw the bot
  --start box size below the current box position
  local start_row = torch.ceil(torch.cmax(torch.ones(1),self.bot_location[1]-self.bot_size))[1]
  local end_row = torch.floor(torch.cmin(torch.zeros(1):add(self.image_size),self.bot_location[1]+self.bot_size))[1]
  local start_col = torch.ceil(torch.cmax(torch.ones(1),self.bot_location[2]-self.bot_size))[1]
  local end_col = torch.floor(torch.cmin(torch.zeros(1):add(self.image_size),self.bot_location[2]+self.bot_size))[1]
  for i=start_row,end_row do
    for j=start_col,end_col do
      local current_location = torch.zeros(2):add(i)
      current_location[2] = j
      if torch.norm(self.bot_location:float() - current_location:float()) < self.bot_size then
        self.screen[{ {1},{i},{j} }] = self.screen[{ {1},{i},{j} }] + .9
      end
    end
  end
  local start_row = torch.ceil(torch.cmax(torch.ones(1),self.goal_location[1]-self.goal_size))[1]
  local end_row = torch.floor(torch.cmin(torch.zeros(1):add(self.image_size),self.goal_location[1]+self.goal_size))[1]
  local start_col = torch.ceil(torch.cmax(torch.ones(1),self.goal_location[2]-self.goal_size))[1]
  local end_col = torch.floor(torch.cmin(torch.zeros(1):add(self.image_size),self.goal_location[2]+self.goal_size))[1]
  for i=start_row,end_row do
    for j=start_col,end_col do
      local current_location = torch.zeros(2):add(i)
      current_location[2] = j
      if torch.norm(self.goal_location:float() - current_location:float()) < self.goal_size then
        self.screen[{ {3},{i},{j} }] = self.screen[{ {3},{i},{j} }] + .9
      end
    end
  end

  self.screen = self.screen - torch.mean(self.screen)
  self.screen = self.screen / torch.sqrt(torch.var(self.screen))
end


function Simulator:score()
  --return the 2 norm - this is a hack that should speed up learning
  return 1 - torch.norm(self.bot_location:float() - self.goal_location:float()) / (torch.sqrt(torch.pow(self.image_size,2)) *2)
end

function Simulator:do_action(action)
  if action == 1 then
    -- go left
    self.bot_location[2] = self.bot_location[2] - 4
  elseif action == 2 then
    -- go right
    self.bot_location[2] = self.bot_location[2] + 4
  elseif action == 3 then
    -- go up
    self.bot_location[1] = self.bot_location[1] - 4
  elseif action == 4 then
    -- go down
    self.bot_location[1] = self.bot_location[1] + 4
  end
    self:check_max()
    self:render()
    if self:did_win() then
        return self.screen,self.reward,true
    end
    return self.screen,self:score(),nil
end


function Simulator:did_win()
  --first check if the bot and goal are overlapping
  local blue_plus_red = self.screen[{ {1} }] + self.screen[{ {3} }]
  if torch.max(blue_plus_red) > 12 then
    --there's an overlap!
    return true
  end
  return false
end

local FakeActor = torch.class('FakeActor')

function FakeActor:__init(num_actions)
  self.number_of_actions = num_actions
end

function FakeActor:return_action(simulator_screen)
  return torch.random(self.number_of_actions)
end
