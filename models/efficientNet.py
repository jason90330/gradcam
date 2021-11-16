from efficientnet_pytorch import EfficientNet, model
from torch import nn
# import config as cfg

class MyEfficientNet(nn.Module):

    def __init__(self, pretrain=True):
        super().__init__()

        # EfficientNet
        if pretrain:
            self.network = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            self.network = EfficientNet.from_name("efficientnet-b0")



        # self.network.set_swish(memory_efficient=False)
        # Replace last layer
        self.network._fc = nn.Sequential(nn.Linear(self.network._fc.in_features, 512), 
                                         nn.ReLU(),  
                                         nn.Dropout(0.25),
                                         nn.Linear(512, 128), 
                                         nn.ReLU(),  
                                         nn.Dropout(0.50), 
                                         nn.Linear(128,2))

    def forward(self, x):
        out = self.network(x)
        return out


