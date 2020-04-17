"""
Replay memory -- or experience buffer -- used to cache a set of recent
experiences while training an egent with TD3/DDPG.

Author: Lulof Pirée
"""
import random

class ReplayMemory():
    """
    Datatype for organizing replay memory. 
   
    Replay memory is an array of 'experiences' 
    (which usually contain a state, action and reward).
    Works as a sort of stack,
    but pushing new things removes oldest,
    and instead of pop a random sampled element is returned
    (and not removed from the datatype).
    Has a fixed capacity for the maximum amount of experiences it can hold.
    """
    
    def __init__(self, capacity):
        """
        Sets up the stack.
        
        Parameters:
        * capacity: integer, maximum number of experiences allowed in the replay
                memory.
        """
        self._capacity = capacity
        # Actual list of experiences:
        self._memory = []
        # Index of oldest element. 
        # When need to discard an element for the first time,
        # it will be the first element of the memory.
        self._current_idx = 0
        
    def can_sample(self, batch_size):
        """
        Checks if it is possible to return a sample.
        
        Parameters:
        * batch_size: integer, amount of experiences returned at once 
                during sampling.
        """
        return (batch_size <= len(self._memory))
    
    def sample(self, batch_size):
        """
        Picks a random sample of [batch_size] experiences from the memory.
        
        Parameters:
        * batch_size: integer, amount of experiences returned at once 
                during sampling.
        
        Returns:
        * sample: list of experiences, of length [batch_size].
                Random samples from the memory.
        """
        assert self.can_sample(batch_size),\
            "ReplayMemory: cannot sample batch, too few experiences"
        return random.sample(self._memory, batch_size)
    
    def push(self, experience):
        """
        Adds new experience to the memory.
        
        If the amount of experiences exceeds the capacity of the ReplayMemory,
        the oldest experience will be discarded to make place for the new.
        
        Parameters:
        * experience: (state, action, reward) tuple (action and reward are 
                integers, state environment-dependend).
        """
        if (len(self._memory) < self._capacity):
            self._memory.append(experience)
        else:
            # Override oldest memory in list.
            self._memory[self._current_idx] = experience
            # Now the next element is the oldest. 
            # Wrap to start when at end of list.
            self._current_idx = (self._current_idx + 1) % self._capacity