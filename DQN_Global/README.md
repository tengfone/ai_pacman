
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

LR = 0.001
Gamma = 0.8
2 Linear Layer, 7056 > 1028 > 64 > 4
Interleaves Explore and Exploitation  

![Linear_V1](./Linear_V1.PNG)

## Linear_V2

![Linear_V2](./Linear_V2.PNG)