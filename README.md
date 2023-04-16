# Multidataset Multitasking

## Description
In this repo, I try to implement a simple neural network (backbone VGG16) in PyTorch that learn 2 different tasks (currently binary classification and multiclass classification on 2 different datasets). The idea is to have a network that learn multiple computer vision tasks, then use its trained weights as pretrained weights for our main task so that it could outperform tradional initialization methods.

## Installation
```
$ pip install -r requirements.py
```

## Dataset
I experiement on two classifcation datasets, you can download them in the following links:
- Pizza or Not Pizza (Binary Classification): [link](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza)
- Wonders of World (Multiclass Classification): [link](https://www.kaggle.com/datasets/karnikakapoor/wonders-of-world)

When download completed, unzip and put them into `./data` folder.

## Training
To train multitask model, run:
```
$ python train.py --type hybrid --dataset pizza_not_pizza wonders_of_world
```
Single task training is also provided. You just simply modify __type__ argument to __single__, then specify the dataset you want to train on (currently support pizza_not_pizza and wonders_of_world dataset):
```
$ python train.py --type single --dataset pizza_not_pizza
```



