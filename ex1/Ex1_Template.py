#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:25:02 2018
"""
#
# put your "import" statements here
#

def MyArcTan(x,N):
    #
    # put your function definition here
    # and make sure you return the correct answer
    #
    return x

MyInput = '0'
while MyInput != 'q':
    MyInput = input('Enter a choice, "a", "b", "c" or "q" to quit: ')
    print('You entered the choice: ',MyInput)

    if MyInput == 'a':
        print('You have chosen part (a)')
        #
        # sample code for part (a)
        # needs to be improved to get better marks
        #
        Input_x = input('Enter a value for x (floating point number): ')
        x = float(Input_x)
        Input_N = input('Enter a value for N (positive integer): ')
        N = int(Input_N)
        print('The answer is: ',MyArcTan(x,N))

    elif MyInput == 'b':
        print('You have chosen part (b)')
        #
        # put your code for part (d) here
        #

    elif MyInput != 'q':
        print('This is not a valid choice')

print('You have chosen to finish - goodbye.')
