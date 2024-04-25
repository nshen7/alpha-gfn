from config import *
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from utils import ops_roll_std, ops_roll_corr


class AlphaGFN():
    '''
    This is a wrapper class for alpha factor generation used for GFlowNet optimization. 
    '''
    def __init__(self):
        
        self.stack: List[pd.DataFrame] # Operand stack; at most two data frames since our operators takes at most two input
        self.state: List[int] # Stores the current state (i.e., integer encoding of token sequence)

        self.action_space: List[str] = ACTION_SPACE
        self.size_action: int = SIZE_ACTION
        self.feature_data: Dict[str, pd.DataFrame] = FEATURE_DATA
        self.max_expr_length: int = MAX_EXPR_LENGTH
        self.window_size: int = WINDOW_SIZE
        
        self._device = DEVICE

    def reset(self): 
        '''
        Reset the operand stack and state (which should be done at the begining of every trajectory)
        '''
        if self.size_action != len(self.action_space):
            raise ValueError
        # self.counter = 0
        self.stack = []
        self.state = [0] # Initial state only contains [BEG] token (encoded as 0)
        
        return     
        
    def get_tensor_state(self) -> Tensor:
        '''
        Remove the BEG token and convert the current state (i.e., self.state) to a padded index tensor 
        that can be input to nn.Embeddings.
        '''
        
        # tensor = nn.functional.one_hot(Tensor(self.state).long(), num_classes=SIZE_ACTION)
        # padding = (0, 0, 0, self.max_expr_length - len(self.state)) # (left, right, top, bottom)
        # padded_tensor = nn.functional.pad(tensor, padding, mode='constant', value=0)
        
        tensor = torch.LongTensor(self.state[1:])
        padding = (0, self.max_expr_length - len(self.state))
        padded_tensor = nn.functional.pad(tensor, padding, mode='constant', value=0)

        return padded_tensor
        
    def _action_to_token(self, action: int) -> str: 
        '''
        Convert an integer-indexed action to its corresponding string token (reference: the section 'Action space').
        
        Input: integer encoding of an action
        Output: string token of that action
        
        action == OFFSET_BEG(0) -> BEG
        action in OFFSET_UNARY:OFFSET_BINARY(1:3) -> unary ops
        action in OFFSET_BINARY:OFFSET_FEATURE(4:8) -> binary ops
        action in OFFSET_FEATURE:OFFSET_SEP(9:13) -> features
        action == OFFSET_SEP(14) -> SEP
        '''
        if action < OFFSET_BEG:
            raise ValueError
        elif action < OFFSET_UNARY:
            return "BEG"
        elif action < OFFSET_BINARY:
            return UNARY[action - OFFSET_UNARY]
        elif action < OFFSET_FEATURE:
            return BINARY[action - OFFSET_BINARY]
        elif action < OFFSET_SEP:
            return FEATURES[action - OFFSET_FEATURE]
        elif action == OFFSET_SEP:
            return "SEP" # the end token
        else:
            assert False  
    
    def get_forward_masks(self) -> Tensor: 
        '''
        Obtain a boolean-valued vector tha indicates the set of masked/valid actions in the forward process.
        
        The general rule is:
            - if the operand stack contains no operand, only features and SEP can be pushed into the stack
            - if the operand stack contains one operand, binary operators can be applied and features can be pushed into the stack
            - if the operand stack contains two operand, only (unary or binary) operators can be applied
        '''
        
        forward_valid_bool = np.zeros(SIZE_ACTION)
        
        if len(self.stack) == 0: # only features and SEP are allowed
            forward_valid_bool[OFFSET_FEATURE:OFFSET_SEP] = 1
        elif len(self.stack) == 1: # only features and unary ops are allowed
            forward_valid_bool[OFFSET_UNARY:OFFSET_BINARY] = 1 
            forward_valid_bool[OFFSET_FEATURE:OFFSET_SEP] = 1
            forward_valid_bool[OFFSET_SEP] = 1
        elif len(self.stack) == 2: # only ops are allowed
            forward_valid_bool[OFFSET_UNARY:OFFSET_FEATURE] = 1
        else:
            raise ValueError("self.stack cannot have more than 2 elements.")
        
        ## If the last action was ops_abs, we forbid another following ops_abs
        if self._action_to_token(self.state[-1]) == "ops_abs":
            forward_valid_bool[self.state[-1]] = 0
            
        return Tensor(forward_valid_bool).bool()

    def get_backward_masks(self) -> Tensor: 
        '''
        Obtain a boolean-valued vector tha indicates the set of masked/valid actions in the backward process.
        In this implementation we only consider the last action given a current state as the only valid action
        '''
        last_action = self.state[-1]
        
        backward_valid_bool = np.zeros(SIZE_ACTION)
        backward_valid_bool[last_action] = 1
        
        return Tensor(backward_valid_bool).bool()
         
        
    def step(self, action: int): 
        '''
        Update the class attributes self.stack and self.state with the newly sampled action
        '''
                
        if action < OFFSET_UNARY or action > OFFSET_SEP:
            ValueError('Invalid action.')

        self.state.append(action)
        if action < OFFSET_BINARY: ## action = unary ops
            token = self._action_to_token(action)
            operand = self.stack.pop()
            if token == "ops_abs":
                res = operand.apply(np.abs)
            elif token == "ops_log":
                res = operand.apply(np.log)
            elif token == "ops_roll_std":
                res = ops_roll_std(operand, window_size=self.window_size)
            else:
                ValueError()
            self.stack.append(res)
                
        elif action < OFFSET_FEATURE: ## action = binary ops
            token = self._action_to_token(action)
            operand_2 = self.stack.pop()
            operand_1 = self.stack.pop()
            if token == "ops_add":
                res = operand_1 + operand_2
            elif token == "ops_subtract":
                res = operand_1 - operand_2
            elif token == "ops_multiply":
                res = operand_1 * operand_2
            elif token == "ops_divide":
                res = operand_1 / operand_2
            elif token == "ops_roll_corr":
                res = ops_roll_corr(operand_1, operand_2, window_size=self.window_size)
            else:
                ValueError()
            self.stack.append(res)
            
        elif action < OFFSET_SEP: ## action = features
            token = self._action_to_token(action)
            self.stack.append(self.feature_data[token])
            
        return