# Model extraction attacks

[door.py](https://github.com/the-nihilist-ninja/dl-model-extraction/blob/master/door.py) has the high-level experiment code 

## About
The aim of the project is to analyse model extraction techniques.  Furthermore, it analyses how the extracted model can be used to do membership inference attack on the original model. It further explores if such an extracted momdel can be used to perform adversarial attacks on the original model

## Experimental setup:
- Victim models for datasets CIFAR-10 and CIFAR-100 were used to carry-out extraction attack analysis. The CIFAR-10 models were taken directly from [here](https://zenodo.org/record/4431043). For CIFAR-100, we trained our own models to act as victim.
- Attacker model architecture was varied 
- The extraction technique was also run on an Out of Distrbution dataset which we put together from downsampled ImageNet data (32x32). An many-to-one relation between the imagenet class and cifar-10 was prepared. Data for classes Deer and Horse were not found in Imagenet, hence they were downloaded from the internet and downsampled to 32 x 32.
-    

## Resources
To help us analyse the performance, we restricted our experiments to CIFAR-10 and CIFAR-100. To emulate the victim model, we used pre-trained models from the following repositories:
- https://zenodo.org/record/4431043

## References
Data Free Model Extraction:
https://arxiv.org/pdf/2011.14779.pdf
