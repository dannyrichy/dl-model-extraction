# Model extraction attacks

[door.py](https://github.com/the-nihilist-ninja/dl-model-extraction/blob/master/door.py) has the high-level experiment code 

## About
The aim of the project is to analyse model extraction techniques.  Furthermore, it analyses how the extracted model can be used to do membership inference attack on the original model. It further explores if such an extracted momdel can be used to perform adversarial attacks on the original model

## Resources
To help us analyse the performance, we restricted our experiments to CIFAR-10 and CIFAR-100. To emulate the victim model, we used pre-trained models from the following repositories:
- https://github.com/chenyaofo/pytorch-cifar-models
- https://zenodo.org/record/4431043

## References
WIP