from torchvision import transforms
from torch.utils.data import ConcatDataset
from dataset.dataset import *

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225)
    )
])

dataset_classes = {
    'wonders_of_world': WondersOfWorldDataset,
    'pizza_not_pizza': PizzaNotPizzaDataset
}

def choose_single_dataset(dataset_name: str, dataset_cfg: dict):
    if dataset_name == 'wonders_of_world':
        task_dataset = WondersOfWorldDataset(
            dataset_cfg[dataset_name]['root_dir'],
            image_transforms
        )
    elif dataset_name == 'pizza_not_pizza':
        task_dataset = WondersOfWorldDataset(
            dataset_cfg[dataset_name]['root_dir'],
            image_transforms
        )

    return task_dataset

def choose_multi_dataset(dataset_name_lst: list, dataset_cfg: dict):
    task_dataset_lst = [
        dataset_classes[dataset_name](
            root_dir=dataset_cfg[dataset_name]['root_dir'],
            input_transforms=image_transforms,
            task_id=idx
        ) for idx, dataset_name in enumerate(dataset_name_lst)
    ]

    concat_dataset = ConcatDataset(task_dataset_lst)

    return concat_dataset 