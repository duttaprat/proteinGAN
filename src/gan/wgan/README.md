# WGAN-GP 

Based on WGAN-GP paper https://arxiv.org/pdf/1704.00028.pdf

## WGAN-GP vs SNGAN

Tests were carried about using MNIST

Given the same generator and discriminator architecture (the same about of trainable parameters):

- SNGAN is faster (1 step/s vs 3.5 steps/s)
- SNGAN losses are more stable from very beginning
- SNGAN can have higher learning rate.
- SNGAN was quicker to get to reasonable state - to the state were numbers can be recognized by a person easily 
(however, not by much - insignificant)
