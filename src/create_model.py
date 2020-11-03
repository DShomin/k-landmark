from efficientnet_pytorch import EfficientNet
from torch import nn
import torch.nn.functional as F
import torch

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class eff_model(nn.Module):
    def __init__(self, model_name, emb_size, do_gem):
        super(eff_model, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name, num_classes=emb_size)
        if do_gem:
            self.model._avg_pooling = GeM()
        self.fc = nn.Linear(emb_size, 1049, bias=False)

    def forward(self, x):
        x = self.model(x)

        x = F.normalize(x, p=2, dim=1)
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)
        x = self.fc(x)
        return x

def build_model(model_name, do_gem):

    model = eff_model(model_name, 1024, do_gem=True)
    # model = EfficientNet.from_pretrained(model_name, num_classes=2048)
    # model._avg_pooling = GeM()
    return model