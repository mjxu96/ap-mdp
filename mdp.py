#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 13:10:10 2018

@author: minjun
"""

# In[]
# Import Package
import mdptoolbox
import numpy as np


# In[]
def calculate_pdf(x, mu, sigma, distribution='Gaussian'):
    ''' Probability Density Function Calculation
    
        Parameter:
            x:              [1 * 1] given x
            mu:             [1 * 1] mean value of distribution
            sigma:          [1 * 1] standard deviation of distribution
            distribution:   [string] name of distribution (default as Gaussian)
        Return:
            probability:    [1 * 1] pdf of a given x of a distribution
    '''  
    probability = 0    
    if distribution == 'Gaussian':
        probability = np.exp(-(x-mu)**2 / (2*(sigma**2))) / (np.sqrt(2*np.pi) * 
                             sigma)
    elif distribution == 'Poission':
        probability = 0
    return probability

# In[]
def collect_point_priority(N, degree=180, file_name=None):
    ''' Collect Priority of Points from File or Random Initialiazation
    
        Parameter:
            N:              [1 * 1] number of points
            degree:         [1 * 1] total degrees that sensor can move (default 
                                    as 180)
            filename:       [string] file name if imported from file
        Return:
            point_priority: [3 * N] (x, y, priority) of N points
    '''
    if file_name == None:
        x_vec = np.random.normal(degree/2, degree/10, (N, 1))
        y_vec = np.random.uniform(0, degree, (N, 1))
        priority_vec = np.random.uniform(0, 1, (N, 1))
        point_priority = np.array([x_vec, y_vec, priority_vec]).reshape((3, N))
    return point_priority

# In[]
def calculate_trans_prob_matrice(sigma=30, degree=180):
    ''' Transition Probability Matrice Calculation
    
        Parameter:
            sigma:          [1 * 1] Gaussian distribution standard deviation 
                                    (default as 30)
            degree:         [1 * 1] total degrees that sensor can move (default 
                                    as 180)
        Return:
            P:              [1 * degree * degree] transition probability matrice 
    '''
    P = np.zeros((1, degree, degree))
    for i in range(degree):
        for j in range(degree):
            P[0][i][j] = calculate_pdf(j, i, sigma, 'Gaussian')
    P[0] = P[0] / np.sum(P[0], axis=0, keepdims=0).reshape(degree, 1)
    return P
    
# In[]
def calculate_reward_matrice(point_priority, degree=180, detect_range=30):
    ''' Reward Matrice Calculation
    
        Parameter:
            point_priority: [3 * number of points] matrice of priority of points
            degree:         [1 * 1] total degrees that sensor can move (default
                                    as 180)
            detect_range:   [1 * 1] the range of sensor that can detect (default 
                                    as 30)
        Return:
            R:              [degree * 1] reward matrice 
    '''
    R = np.zeros((degree, 1))
    for index in range(degree):
        temp = 0
        temp = np.sum(((point_priority[0][:] < (index+detect_range/2)) * 
                        (point_priority[0][:] > (index-detect_range/2))) * 
                        point_priority[2][:])
        R[index][0] = temp
    return R

# In[]
def calculate_policy(P, R):
    ''' Policy Calculation Function using Value Iteration
    
        Parameter:
            P:              [1 * degree * degree] transition probability matrice
            R:              [degree * 1] reward matrice 
        Return:
            V:              [degree * 1] value function of given states S
    '''
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    vi.setSilent()
    vi.run()
    V_arr = np.array([vi.V]).T
    return V_arr

# In[]
def test():
    P = calculate_trans_prob_matrice()
    point_priority = collect_point_priority(10)
    R = calculate_reward_matrice(point_priority)
    V = calculate_policy(P, R)
    pos = np.where(V == np.max(V))
    print(pos[0][0])
    

    

    
    
    