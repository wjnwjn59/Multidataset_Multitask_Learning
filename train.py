import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import argparse
import yaml
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from networks.VGGHybridNet import VGGHybridNet
from networks.SingleVGGNet import SingleVGGNet

from utils.create_dataset import (
    choose_single_dataset, 
    choose_multi_dataset
)

def train_singletask(
    model, 
    dataset,
    optimizer, 
    #scheduler,
    criterion, 
    num_epochs, 
    device
):
    train_dataloader, val_dataloader = dataset
    for epoch in range(num_epochs):
        model.train()
        
        train_loss = 0.0
        running_items = 0
        for idx, (inputs, labels) in enumerate(train_dataloader):
            images, ids = inputs
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()
            
            running_items += images.size(0)
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / running_items
        
        model.eval()

        val_loss = 0.0
        running_items = 0
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_dataloader):
                images, ids = inputs
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)

                loss = criterion(outputs, labels)

                running_items += images.size(0)
            
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / running_items
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


def train_multitask(
    model, 
    dataset,
    optimizer, 
    #scheduler,
    criterions, 
    num_epochs, 
    device
):
    
    train_dataloader = dataset[0]
    val_dataloader = dataset[1]
    
    task1_criterion = criterions[0]
    task2_criterion = criterions[1]

    for epoch in range(num_epochs):
        model.train()
        
        train_loss = 0.0
        running_items = 0
        for idx, (inputs, labels) in enumerate(train_dataloader):
            images, ids = inputs
            images = images.to(device)
            labels = labels.to(device)

            task1_mask = ids == 0
            task2_mask = ids == 1

            task1_loss = torch.tensor(0)
            task2_loss = torch.tensor(0)

            optimizer.zero_grad()

            if task1_mask.any():
                task1_images = images[task1_mask]
                task1_labels = labels[task1_mask]
                task1_outputs = model(task1_images, task=0)

                task1_loss = task1_criterion(task1_outputs, task1_labels)

            if task2_mask.any():
                task2_images = images[task2_mask]
                task2_labels = labels[task2_mask]
                task2_outputs = model(task2_images, task=1)

                task2_loss = task2_criterion(task2_outputs, task2_labels)
            
            loss = task1_loss * 0.5 + task2_loss * 0.5

            loss.backward()

            optimizer.step()

            #scheduler.step()
            
            running_items += images.size(0)
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / running_items
        
        model.eval()

        val_loss = 0.0
        running_items = 0
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_dataloader):
                images, ids = inputs
                images = images.to(device)
                labels = labels.to(device)

                task1_mask = ids == 0
                task2_mask = ids == 1

                task1_loss = torch.tensor(0)
                task2_loss = torch.tensor(0)
                
                if task1_mask.any():
                    task1_images = images[task1_mask]
                    task1_labels = labels[task1_mask]
                    task1_outputs = model(task1_images, task=0)

                    task1_loss = task1_criterion(task1_outputs, task1_labels)

                if task2_mask.any():
                    task2_images = images[task2_mask]
                    task2_labels = labels[task2_mask]
                    task2_outputs = model(task2_images, task=1)

                    task2_loss = task2_criterion(task2_outputs, task2_labels)
                
                loss = task1_loss * 0.5 + task2_loss * 0.5

                running_items += images.size(0)
            
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / running_items
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--type', 
    	type=str,
        required=True,
        choices=['single', 'hybrid'],
    	help='Type of network to train'
    )
    parser.add_argument(
    	'--dataset', 
    	nargs='+',
        required=True,
    	help='Specify the dataset to train'
    )
    parser.add_argument(
    	'--epochs', 
    	type=int, 
        default=50,
    	help='number of iteration'
    )
    parser.add_argument(
    	'--learning_rate', 
    	type=float, 
        default=1e-5,
    )
    parser.add_argument(
    	'--batch_size', 
    	type=int, 
        default=64,
    )
    parser.add_argument(
    	'--config_path', 
    	type=str, 
        default='./cfg/config.yml',
    	help='path to the hyperparameter config file'
    )
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    dataset_cfg = CONFIG['dataset']
    train_cfg = CONFIG['train']

    if args.type == 'single':
        task_dataset = choose_single_dataset(args.dataset[0], dataset_cfg)
    elif args.type == 'hybrid':
        task_dataset = choose_multi_dataset(args.dataset, dataset_cfg)

    train_size = int(len(task_dataset) * train_cfg['train_size'])
    val_size = int(len(task_dataset) * train_cfg['val_size'])
    test_size = len(task_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        task_dataset, [train_size, val_size, test_size]
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=train_cfg['test_batch_size']
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=train_cfg['test_batch_size']
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.type == 'single':
        model = SingleVGGNet(
            dataset_cfg[args.dataset[0]]['n_classes'],
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.learning_rate
        )
        # scheduler = optim.lr_scheduler.StepLR(
        #     optimizer, 
        #     step_size=10, 
        #     gamma=0.1
        # )
        criterion = nn.CrossEntropyLoss()

        train_singletask(
            model=model,
            dataset=(train_dataloader, val_dataloader),
            optimizer=optimizer,
            #scheduler=scheduler,
            criterion=criterion,
            num_epochs=args.epochs,
            device=device
        )
    elif args.type == 'hybrid':
        model = VGGHybridNet(
            dataset_cfg[args.dataset[0]]['n_classes'],
            dataset_cfg[args.dataset[1]]['n_classes']
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=train_cfg['weight_decay']
        )
        # scheduler = optim.lr_scheduler.StepLR(
        #     optimizer, 
        #     step_size=10, 
        #     gamma=0.1
        # )

        criterion_1 = nn.CrossEntropyLoss()
        criterion_2 = nn.CrossEntropyLoss()

        train_multitask(
            model=model,
            dataset=(train_dataloader, val_dataloader),
            optimizer=optimizer,
            #scheduler=scheduler,
            criterions=(criterion_1, criterion_2),
            num_epochs=args.epochs,
            device=device
        )
    
if __name__ == '__main__':
    main()