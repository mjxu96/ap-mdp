#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:18:43 2018

@author: minjun
"""

import mdp

if __name__ == '__main__':
    P = mdp.calculate_trans_prob_matrice()
    priority = mdp.collect_point_priority(200)
    R = mdp.calculate_reward_matrice(priority)
    V = mdp.calculate_policy(P, R)
    