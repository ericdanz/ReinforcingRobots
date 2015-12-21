require 'Simulator'
require 'torch'
require 'sys'

local class = require 'class'

torch.setnumthreads(4)

function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end


-- Get the mean value of a table
function mean( t )
  local sum = 0
  local count= 0

  for k,v in pairs(t) do
    if type(v) == 'number' then
      sum = sum + v
      count = count + 1
    end
  end

  return (sum / count)
end


function make_one_set(simulator,actor,epsilon,number_of_steps)
  local state_table = {}
  local state = {}
  for i=1,number_of_steps do
    local action = nil
    if torch.uniform() < epsilon then
      --do a random action
      action = torch.random(actor.number_of_actions)
    else
      action = actor:return_action(simulator.screen)
    end
      state['action'] = action
      state['screen'],state['score'],did_win = simulator:do_action(action)
      
      state_table[#state_table+1] = deepcopy(state)

  end

      if did_win then
        local reward = state['score']
        local discount_factor = 0.1
        for i=0,#state_table-1 do
          --going backwards
          state_table[#state_table-i]['score'] = reward*torch.pow(discount_factor,#state_table-i)
        end
      end

      return state_table
end

function make_states(simulator,actor,epsilon,number_of_steps,number_of_games,winners_only)
  local winners = winners_only or false
  local game_table = {}
  local length_table = {}
  for i=1,number_of_games do
      simulator:reset(simulator.image_size,10)
      local state_table = make_one_set(simulator,actor,epsilon,number_of_steps)
      --tying the mix of wins to epsilon allows a decay in random games, leading towards
      --more wins as the learning progresses
      while (#state_table == number_of_steps) and (winners) and (torch.uniform() > (math.min(0,epsilon) + 0.1) ) do
          state_table = make_one_set(simulator,actor,epsilon,number_of_steps)
          print('doing extra')
        end
      length_table[#length_table + 1] = #state_table
      for j=1,#state_table do
        game_table[#game_table+1] = state_table[j]
      end
    end
  print("The average game length (lower is better, and 10 is the max): "..tostring(mean(length_table)))
  return game_table,mean(length_table)
end


local faker = FakeActor(4)
local sim = Simulator(64,10)
local start_time = os.clock()
local returned_states = make_states(sim,faker,1,10,100,false)
print(os.clock()-start_time)
require 'image'
for i=1,10 do
  print(i)
  --_win = image.display{image=returned_states[i]['screen'],win=_win}
  sys.sleep(0.3)
end
