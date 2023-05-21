import numpy as np
from world import World


class DynamicProgramming:

    def __init__(self):
        self.V_s = None # Will store the value solution table
        self.Q_sa = None # Will store the action-value solution table
        
    def value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Value Iteration (VI)")
        # Initialize value table
        V_s = np.zeros(env.n_states)
        
        max_iter = 10_000
        for i in range(max_iter):
            delta = 0
            V_new = np.zeros(env.n_states)

            for s in env.states:
                max_val = 0
                
                for i_action, a in enumerate(env.actions):
                    tf = env.transition_function(s,a)
                    reward = tf[1]
 
                    for s_next in env.states:
                        if s_next == tf[0]:
                            # If the next state is the one we transitioned to
                            reward += gamma * V_s[s_next]
                    
                    max_val = max(max_val, reward)
                # Update values for state s
                V_new[s] = max_val
                delta = max(delta, abs(V_s[s] - V_new[s]))

            V_s = V_new
            print("delta: ", delta)
            if delta < theta:
                break
        self.V_s = V_s
        return

    def Q_value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes Q-value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Q-value Iteration (QI)")
        # Initialize state-action value table
        Q_sa = np.zeros([env.n_states,env.n_actions])

        max_iter = 10_000
        for i in range(max_iter):
            delta = 0
            Q_new = np.zeros([env.n_states,env.n_actions])

            for s in env.states:
                for i_action, a in enumerate(env.actions):
                    max_val = 0
                    
                    tf = env.transition_function(s,a)
                    reward = tf[1]
 
                    for s_next in env.states:
                        if s_next == tf[0]:
                            # If the next state is the one we transitioned to
                            reward += gamma * np.max(Q_sa[s_next])
                    
                    # Update values for state/action combination
                    Q_new[s][i_action] = reward
                    delta = max(delta, abs(Q_sa[s][i_action] - Q_new[s][i_action]))

            Q_sa = Q_new

            if delta < theta:
                break
        self.Q_sa = Q_sa
        return
           
    def execute_policy(self,env,table='V'):
        # Execute the greedy action, starting from the initial state
        env.reset_agent()
        print("Start executing. Current map:") 
        env.print_map()
        while not env.terminal:
            current_state = env.get_current_state() # this is the current state of the environment, from which you will act
            available_actions = env.actions
            # Compute action values
            if table == 'V' and self.V_s is not None:
                # Save action/reward combinations for current state in a list
                actions = []
                for a in env.actions:
                    tf = env.transition_function(env.get_current_state(), a)
                    actions.append(tuple((a, self.V_s[tf[0]])))
                
                goal_action = list(a for a in actions if a[1] == 0.0)
                if goal_action:
                    # If there is a goal action, execute it
                    greedy_action = goal_action[0][0]
                else:
                    # Else, execute the action with the highest reward
                    greedy_action = max(actions, key=lambda item:item[1])[0]
            elif table == 'Q' and self.Q_sa is not None:
                current_state = env.get_current_state()
                s_a_values = self.Q_sa[current_state]
                i_action = get_greedy_index(s_a_values)[0][0]
                # Execute the action with the highest reward
                greedy_action = env.actions[i_action]
            else:
                print("No optimal value table was detected. Only manual execution possible.")
                greedy_action = None


            # Ask the user what he/she wants
            while True:
                if greedy_action is not None:
                    print('Greedy action = {}'.format(greedy_action))    
                    your_choice = input('Choose an action by typing it in full, then hit enter. Just hit enter to execute the greedy action:')
                else:
                    your_choice = input('Choose an action by typing it in full, then hit enter. Available are {}'.format(env.actions))
                    
                if your_choice == "" and greedy_action is not None:
                    executed_action = greedy_action
                    env.act(executed_action)
                    break
                else:
                    try:
                        executed_action = your_choice
                        env.act(executed_action)
                        break
                    except:
                        print('{} is not a valid action. Available actions are {}. Try again'.format(your_choice,env.actions))
            print("Executed action: {}".format(executed_action))
            print("--------------------------------------\nNew map:")
            env.print_map()
        print("Found the goal! Exiting \n ...................................................................... ")
    

###
# Helper functions
###

def get_greedy_index(action_values):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max. 
    Optional to uses '''
    return np.where(action_values == np.max(action_values))
