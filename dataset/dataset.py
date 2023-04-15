import os

from PIL import Image
from torch.utils.data import Dataset

class WondersOfWorldDataset(Dataset):
    def __init__(
            self, 
            root_dir,
            input_transforms=None,
            task_id=0
        ):

        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.input_transforms = input_transforms
        self.task_id = task_id
        self.label_encoder = {
            class_name: i for i, class_name in enumerate(self.classes)
        }
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                sample = (filepath, self.label_encoder[class_name])
                samples.append(sample) 
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        
        image = Image.open(filepath).convert('RGB')
        
        if self.input_transforms:
            image = self.input_transforms(image)
        
        return (image, self.task_id), label

class PizzaNotPizzaDataset(Dataset):
    def __init__(
            self, 
            root_dir,
            input_transforms=None,
            task_id=1
        ):

        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.input_transforms = input_transforms
        self.task_id = task_id
        self.label_encoder = {
            class_name: i for i, class_name in enumerate(self.classes)
        }
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                sample = (filepath, self.label_encoder[class_name])
                samples.append(sample) 
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        
        image = Image.open(filepath).convert('RGB')
        
        if self.input_transforms:
            image = self.input_transforms(image)
        
        return (image, self.task_id), label