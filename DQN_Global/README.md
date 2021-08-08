
# Deep Q Learning - Global Window

## CNN_V1
Unable to execute as too many parameters
LR = 0.00001
Gamma = 0.8
Conv2d(7,1)
BatchNorm2d(1)
Linear(945,4)
First 20 games explore, after that is exploit


## Linear_V1

LR = 0.00001

Gamma = 0.8

2 Linear Layer, Input 36 (Global State), Hidden 256, Output 4

First 20 games explore, after that is exploit


## Linear_V2

LR = 0.0001

Gamma = 0.8

2 Linear Layer, Input 28 (Number of columns), Hidden 128, Output 4

First 20 games explore, subsequent games 20% chance explore, 80% chance exploit
