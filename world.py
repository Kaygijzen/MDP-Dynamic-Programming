import numpy as np
import itertools

class World:

    def __init__(self, filename):
        ''' Initializes a world object '''
        # read the map from a txt into self.map_np
        self.map_np,self.dims = read_txt_to_map(filename)

        # Identify state and action space
        self.state_vector_list, self.number_of_keys = get_unique_states_list(self.map_np) # state list as vector representation
        self.states = np.arange(0,len(self.state_vector_list),1) # state list as indices
        self.n_states = len(self.states)
        self.actions = np.array(['up','down','left','right'])
        self.n_actions = len(self.actions)    
        
        # Set current state
        self.start_state = self.find_start_state(self.number_of_keys)
        self.reset_agent()

        print('Initialization map:')
        self.print_map()

    def find_start_state(self,number_of_keys):
        ''' finds the start state of the agent, indicated by * on the start map '''
        location = np.array(np.where(self.map_np == '*')).squeeze()
        self.map_np[location[0],location[1]] = ' ' # remove the agent location from the map
        start_state = np.append(location,np.zeros(number_of_keys,dtype='int'))
        return start_state

    def reset_agent(self):
        ''' sets the agent back to the start position given in the initial map '''
        self.current_state_vector = self.start_state
        self.terminal = False

    def get_current_state(self):
        ''' returns the current agent location '''
        return self._state_vector_to_state(self.current_state_vector)

    def act(self,a):
        ''' transitions and actually updates the internal agent state ''' 
        # find next state and reward
        current_state = self._state_vector_to_state(self.current_state_vector)
        s_prime,r = self.transition_function(current_state,a)
        # update the internal state
        self.current_state_vector = self._state_to_state_vector(s_prime)
        # check if we have terminated        
        if self.map_np[self.current_state_vector[0],self.current_state_vector[1]].isdigit():
            self.terminal = True
        return s_prime, r

    def print_state(self,s):
        ''' Explains what a certain discrete state actually represents '''
        state_vector = self._state_to_state_vector(s)
        location = state_vector[0:2]
        key_booleans = state_vector[2:]
        if len(key_booleans) > 0:
            key_dict = {index_to_character(i+1):bool(key_booleans[i]) for i in range(len(key_booleans))}
        print('State {} refers to: agent location {} and key posession: {}'.format(s,location,sorted(key_dict.items())))

    def print_map(self):
        ''' Makes a nice print of the map, with the current agent position '''
        print_map = np.copy(self.map_np)
        print_map[self.current_state_vector[0],self.current_state_vector[1]] = '*'
        for row in print_map:
            print("  ".join(word for word in row))

    def transition_function(self,s,a):
        ''' Given a state s and action a, this returns [s',R(s,a,s')], i.e.
        the next state and associated reward.
        Note that the environment is deterministic, so each (s,a) has only one
        next state s', and the probability of this observation is always 1.
        We therefore do not pass the transition probabilities '''
        
        state_vector = self._state_to_state_vector(s)
        # Current location
        location = state_vector[0:2] # selects first two elements
        
        # First check whether we are at a terminal state:
        current_location_element = self.map_np[location[0],location[1]]
        if current_location_element.isdigit():
            # we are at a goal, return the same state and a reward of 0
            return s, 0
        
        # Move the agent:
        s_prime = np.copy(state_vector)         
        new_location = np.copy(location)
        if a == 'up':
            new_location[0] -=1 # first coordinate decreases to move up 
        elif a == 'down':
            new_location[0] +=1        
        elif a == 'left':
            new_location[1] -=1 # second coordinate decreases to move left         
        elif a == 'right':
            new_location[1] +=1        
        else:
            raise ValueError('Invalid action specified: {}'.format(a))
        
        # check whether the new location is valid:
        new_location_element = self.map_np[new_location[0],new_location[1]]
        if new_location_element == '#':
            # cannot move there
            new_location = location
        elif new_location_element.isupper():
            # stepped onto a door, check whether we can open it:
            index = value_of_capital_letter(new_location_element) # index of door
            if not state_vector[index+1]:
                # We don't have the key, cannot move there
                new_location = location     

        # update the location in the state:
        s_prime[0:2] = new_location
        
        # check whether we pick up a key:
        if new_location_element.islower():
            index = value_of_lower_letter(new_location_element) # index of key
            s_prime[index+1] = True # We now have this key

        # Check the reward:
        if new_location_element.isdigit():
            # Found the goal
            r = int(new_location_element)*10
        else:
            # every step a small penalty
            r = -1

        return self._state_vector_to_state(s_prime),r

    def _state_vector_to_state(self,state_vector):
        ''' given a state vector, returns the state, i.e., 
        the index in self.state_vector_list that matches state_vector'''
        return np.where(np.all(state_vector == self.state_vector_list,axis=1))[0][0]

    def _state_to_state_vector(self,state):
        ''' returns the underlying state vector for a state index '''
        return self.state_vector_list[state]


###
# Helper functions for indices to letters  
###
              
def value_of_capital_letter(letter):
    ''' Gives an index to a capital letter, i.e. 'A' -> 1, 'B' -> 2 '''
    return ord(letter) - ord('A') + 1

def value_of_lower_letter(letter):
    ''' Gives an index to a capital letter, i.e. 'a' -> 1, 'b' -> 2 '''
    return ord(letter) - ord('a') + 1

def index_to_character(index):
    ''' Turns index into a small lettter, i.e., 1 -> 'a', 2 -> 'b' '''
    return chr(index + 96)

###
# Helper functions for initialization of world
###
    
def read_txt_to_map(filename):
    # Reads a txt file of the world to a numpy map 
    with open(filename) as f:
        txt = [line.rstrip() for line in f]   
    dims = [len(txt),len(txt[0])]
    map_np = np.empty(dims,dtype="<U1").astype('str')
    for i in range(dims[0]):
        for j in range (dims[1]):
            map_np[i,j] = txt[i][j]
    return map_np,dims

def get_unique_states_list(map_np):
    ## Generates a list of unique states in the map
    # Count all free_locations and the number of keys
    dims = map_np.shape
    free_locations = []
    number_of_keys = 0
    for i in range(dims[0]):
        for j in range (dims[1]):
            if map_np[i,j] != '#':
                free_locations.append([i,j]) # store all free locations
            if map_np[i,j].islower():
                number_of_keys += 1 # counted a key
    free_locations = np.array(free_locations)
    
    # Make all unique combinations of having/not having the keys
    if number_of_keys > 0:
        possible_key_combinations = np.array(list(itertools.product([0, 1], repeat=number_of_keys)))
    else:
        possible_key_combinations = None

    # Make a list of all possible unique states. Each state is a combination of a
    # free location and a possible combination of keys. The total number of states
    # equals 'number of free locations' x (2)^'number_of_keys'
    unique_states = []
    for free_location in free_locations:
        if possible_key_combinations is not None:
            for key_combination in possible_key_combinations:                
                unique_states.append(np.concatenate([free_location,key_combination],0))
        else:
            unique_states.append(free_location)
    print('Identified {} free locations and {} keys, leading to {} unique states'.format(
            len(free_locations),number_of_keys,len(unique_states)))
    return unique_states, number_of_keys
