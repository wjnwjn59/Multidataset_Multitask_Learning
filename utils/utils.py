import torch.nn as nn

def unfreeze_pretrained(
        model, 
        is_all=True,
        n_last_layers=3
    ):
    modules = list(model.children())

    if is_all:
        for i in range(len(modules)-1, -1, -1):
            if isinstance(modules[i], (nn.Conv2d, nn.Linear)):
                for param in modules[i].parameters():
                    param.requires_grad = True
    else:
        for i in range(len(modules)-1, -1, -1):
            if isinstance(modules[i], (nn.Conv2d, nn.Linear)) and (len(modules)-i) <= n_last_layers:
                for param in modules[i].parameters():
                    param.requires_grad = True
            elif isinstance(modules[i], nn.MaxPool2d):
                break
    
    return model