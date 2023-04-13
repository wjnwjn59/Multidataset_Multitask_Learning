import torch.nn as nn

def unfreeze_pretrained(
        model, 
        n_last_layers=0
    ):
    layers_to_unfreeze = list(model.features.children())[n_last_layers:]

    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
            
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight.data)
            if layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight.data, 1)
            nn.init.constant_(layer.bias.data, 0)