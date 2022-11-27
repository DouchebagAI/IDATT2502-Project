# Combining Monte Carlo Tree Search & Neural networks to play GO

## Abstract

This assignment uses Monte Carlo Tree Search combined with convolutional neural networks to play the board game Go. AlphaGo, the worlds best Go AI was the main inspiration. Our process had four main steps; Standalone Monte Carlo Tree Search, neural network for best move prediction, neural network for prediction of winner from game state, storing trained models.

By using value network with a tree, we see that it has a significantcly faster decision speed and is slightly better than all other variants.

Our results open up for discussion. As a cause of lack of data-power, we were not able to test on large board sizes. We also have limited power to simulate games when expanding, this might lead to bad or incorrect training data.

A pattern that appears is that a more trained model will perform better than a less trained one. Also, a player that uses a value network will play significantly faster than a player that does not use a value network

# Installation and setup

```bash
# In the root directory
pip install -e .
```
