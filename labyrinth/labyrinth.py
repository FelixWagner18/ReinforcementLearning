#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:08:49 2018

@author: felix
"""

"""
____________
|  _________
|  ______|o|
|  |_____  |
___________|

 1  2  3  4
 5  6  7  8
 9 10 11 12
13 14 15 16 

up = 0
right = 1
down = 2
left = 3
"""

"""
TODO:
    array instead of list 
    check game again, is there a bug?
"""

import numpy as np
from random import *
import operator
import matplotlib.pyplot as plt

def readcodes():
    action_dict = {}
    with open("actioncodes","r") as f:
        actioncodes = f.readlines()
        for i in range(64):
            key, val = actioncodes[i].split()
            action_dict[int(key)] = int(val)    
    return action_dict 

def actionnumb(current_pos, direction):
    if current_pos == 0:
        actionnumber = 0
    elif current_pos == 100:
        actionnumber = 100
    else:    
        actionnumber = current_pos + direction * 16
    return actionnumber

def newpos(actionnumber, action_dict):
    if action_dict[actionnumber] == 0:
        new_position = 0
    elif action_dict[actionnumber] == 1:
        if actionnumber < 17:
            new_position = (actionnumber - 1) % 16 + 1 -4 
        elif actionnumber < 33:
            new_position = (actionnumber - 1) % 16 + 1 +1
        elif actionnumber < 49: 
            new_position = (actionnumber - 1) % 16 + 1 +4
        else: 
            new_position = (actionnumber - 1) % 16 + 1 -1
    elif action_dict[actionnumber] == 2:
        new_position = 100
    return new_position

def greedy(position, action_values):
    direction_values = {}
    direction_values[0] = action_values[actionnumb(position, 0)]
    direction_values[1] = action_values[actionnumb(position, 1)]
    direction_values[2] = action_values[actionnumb(position, 2)]
    direction_values[3] = action_values[actionnumb(position, 3)]
    greedy_direction = max(direction_values.items(), key=operator.itemgetter(1))[0]
    return greedy_direction

def policy(current_pos, action_values, epsilon):
    if random() > epsilon:
        direction = greedy(current_pos, action_values)
        is_greedy = 1
    else:
        direction = randint(0,3)
        is_greedy = 1
    return direction, is_greedy

def update_action_val(new_position, reward, current_pos, actionnumber, action_values, \
                      alpha, gamma):
    if new_position == 0:
        new_greedy_directon = greedy(current_pos, action_values)
        new_greedy_action = actionnumb(current_pos, new_greedy_directon)
    elif new_position == 100:
        new_greedy_directon = greedy(current_pos, action_values)
        new_greedy_action = actionnumb(current_pos, new_greedy_directon)
    else:
        new_greedy_directon = greedy(new_position, action_values)
        new_greedy_action = actionnumb(new_position, new_greedy_directon)
    updated_value = (1-alpha)*action_values[actionnumber] + \
        alpha*(reward + gamma *(action_values[new_greedy_action]))
    return updated_value

def numb_to_dir(number):
    if number == 0:
        direction = 'u'
    elif number == 1:
        direction = 'r'
    elif number == 2:
        direction = 'd'
    elif number == 3:
        direction = 'l'
    return direction
    
action_dict = readcodes()
start_pos = 8

epsilon_start = 1  #policy epsilon greedy
alpha = 0.5  #learning rate
gamma = 0.7  #discounting factor

action_values = {} #initialize Q table to zero
for i in range(64): 
    action_values[i+1] = 0
action_values[0] = 0
action_values[100] = 0

round_count = []
learn_curve = []

episodes = 30
rounds = 500

for j in range(episodes):
    #print("EPISODE ", j)
    epsilon = epsilon_start * (episodes - j)/episodes
    #alpha = 0.5 * (episodes - j)/episodes
    #print ("epsilon: ", epsilon)
    rounds_to_win = rounds
    current_pos = start_pos
    for i in range(rounds): #after max 200 rounds agent should be able to get out
        #print("position: ", current_pos)
        direction, is_greedy = policy(current_pos, action_values, epsilon)
        #print("direction: ", direction)
        actionnumber = actionnumb(current_pos, direction)
        #print("actionnumber: ", actionnumber)
        new_position = newpos(actionnumber, action_dict)
        #print("new_position: ", new_position)
        #print(i, "new_position: ", new_position)    
        if new_position == 0:
            reward = - 1
            if is_greedy == 1: 
                action_values[actionnumber] = update_action_val(new_position, reward,\
                         current_pos, actionnumber, action_values, alpha, gamma)
            #print("updated value: ", action_values[actionnumber])
            #print("wall")
        elif new_position == 100:
            reward = 100
            if is_greedy == 1:
                action_values[actionnumber] = update_action_val(new_position, reward,\
                         current_pos, actionnumber, action_values, alpha, gamma)
            rounds_to_win = i
            #print("updated value: ", action_values[actionnumber])
            break
        else: 
            reward = 0
            if is_greedy == 1:
                action_values[actionnumber] = update_action_val(new_position, reward,\
                         current_pos, actionnumber, action_values, alpha, gamma)
            current_pos = new_position
            #print("updated value: ", action_values[actionnumber])
    #print("rounds to win: ", rounds_to_win)
    round_count.append(np.array(j))
    learn_curve.append(np.array(rounds_to_win))

plt.plot(round_count, learn_curve)
plt.savefig('learning_curve.eps', format='eps')

up_val = np.zeros(16)
right_val = np.zeros(16)
down_val = np.zeros(16)
left_val = np.zeros(16)

for i in range (16):
    up_val[i] = action_values[i+1]
    right_val[i] = action_values[i+1+16]
    down_val[i] = action_values[i+1+32]
    left_val[i] = action_values[i+1+48]
    
up_val = np.resize(up_val, (4,4))
right_val = np.resize(right_val, (4,4))
down_val = np.resize(down_val, (4,4))
left_val = np.resize(left_val, (4,4))    

fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2, 2)
ax0.imshow(up_val, cmap=plt.cm.Blues, aspect='auto')
ax1.imshow(up_val, cmap=plt.cm.Blues, aspect='auto')
ax2.imshow(up_val, cmap=plt.cm.Blues, aspect='auto')
ax3.imshow(up_val, cmap=plt.cm.Blues, aspect='auto')
ax0.set_title('Up')
ax1.set_title('Right')
ax2.set_title('Down')
ax3.set_title('Left')
for i in range(4):
    for j in range(4):
        c0 = round(up_val[j,i],1)
        c1 = round(right_val[j,i],1)
        c2 = round(down_val[j,i],1)
        c3 = round(left_val[j,i],1)
        ax0.text(i, j, str(c0), va='center', ha='center')
        ax1.text(i, j, str(c1), va='center', ha='center')
        ax2.text(i, j, str(c2), va='center', ha='center')
        ax3.text(i, j, str(c3), va='center', ha='center')
plt.savefig('direction_values.eps', format='eps')

greedy_actions = []
for pos in range(16):
    greedy_actions.append(greedy(pos+1, action_values))

greedy_array = np.resize( np.array(greedy_actions) , (4,4) )

fig, ax = plt.subplots()
ax.imshow(greedy_array)
ax.set_title('Greedy Actions')
k = 0
for i in range(4):
    for j in range(4):
        c = numb_to_dir(greedy_actions[i + j*4])
        ax.text(i, j, str(c), va='center', ha='center')
        k = k+1
plt.savefig('greedy_directions.eps', format='eps')
