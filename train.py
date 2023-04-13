import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import argparse
import yaml
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataset.dataset import *
from networks.VGGHybridNet import *

def train(
        model, 
        dataset,
        optimizer, 
        scheduler,
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

            scheduler.step()
            
            running_items += inputs.size(0)
            train_loss += loss.item() * inputs.size(0)
            
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

                    print('Task 1: ', task1_loss)

                if task2_mask.any():
                    task2_images = images[task2_mask]
                    task2_labels = labels[task2_mask]
                    task2_outputs = model(task2_images, task=1)

                    task2_loss = task2_criterion(task2_outputs, task2_labels)
                
                loss = task1_loss * 0.5 + task2_loss * 0.5

                running_items += inputs.size(0)
            
                val_loss += loss.item() * inputs.size(0)
                
        val_loss = val_loss / running_items
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--epochs', 
    	type=int, 
        default=50,
    	required=True, 
    	help='number of iteration'
    )
    parser.add_argument(
    	'--learning_rate', 
    	type=float, 
        default=1e-5,
    	required=True, 
    )
    parser.add_argument(
    	'--batch_size', 
    	type=int, 
        default=64,
    	required=True, 
    )
    parser.add_argument(
    	'--config_path', 
    	type=str, 
        default='./cfg/config.yml',
    	required=True, 
    	help='path to the hyperparameter config file'
    )
    args = parser.parse_args()


    with open(args, 'r') as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset_cfg = CONFIG['dataset']
    train_cfg = CONFIG['train']

    task1_dataset = Task1Dataset(
        dataset_cfg['wonders_of_world']['root_dir'],
        image_transforms
    )
    task2_dataset = Task2Dataset(
        dataset_cfg['pizza_not_pizza']['root_dir'],
        image_transforms
    )
    concat_dataset = ConcatDataset([task1_dataset, task2_dataset])

    train_size = int(len(concat_dataset) * train_cfg['train_size'])
    val_size = int(len(concat_dataset) * train_cfg['val_size'])
    test_size = len(concat_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        concat_dataset, [train_size, val_size, test_size]
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

    model = VGGHybridNet(
        dataset_cfg['wonders_of_world']['n_classes'],
        dataset_cfg['pizza_not_pizza']['n_classes']
    ).to(device)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=train_cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10, 
        gamma=0.1
    )

    train(
        model=model,
        dataset=(train_dataloader, val_dataloader),
        optimizer=optimizer,
        scheduler=scheduler,
        criterions=(criterion_1, criterion_2),
        num_epochs=args.epochs,
        device=device
    )
    
if __name__ == '__main__':
    main()