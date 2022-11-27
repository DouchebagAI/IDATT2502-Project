# Combining Monte Carlo Tree Search & Neural networks to play GO

## Abstract

The goal of the assignment was to use Monte Carlo Tree Search combined with convolutional neural networks to play the board game Go. AlphaGo, the worlds best Go AI was the main inspiration. The method had four main steps; Standalone Monte Carlo Tree Search, neural network for best move prediction, neural network for prediction of winner from game state and storing trained models. 

The results open up for discussion. As a cause of lack of data-power, we were not able to test on large board sizes. We also have limited power to simulate games when expanding, this might lead to bad or incorrect training data and create overfitting. 

Given the limited resources it is hard to conclude with much. However, the model improves with training, a value model reduces the time it takes to take a turn. Also, MSE is the best loss function for this use case and two layers of 2D convolution perfoms better than a single layer.