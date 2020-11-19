from efficientnet_pytorch import EfficientNet
from torch import nn
import torch.nn.functional as F
import torch
import math
import geffnet

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

class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine 

class eff_model(nn.Module):
    def __init__(self, model_name, emb_size, do_gem):
        super(eff_model, self).__init__()
        self.enet = geffnet.create_model(model_name.replace('-', '_'), pretrained=True)
        self.feat = nn.Linear(self.enet.classifier.in_features, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, 1049)
        self.enet.classifier = nn.Identity()
        if do_gem:
            self.enet.global_pool = GeM()

        # self.model = EfficientNet.from_pretrained(model_name)
        
        # # self.fc = nn.Linear(emb_size, 1049, bias=False)
        # self.feat = nn.Linear(1000, emb_size)
        # self.swish = Swish_module()
        # self.metric_classify = ArcMarginProduct_subcenter(emb_size, 1049)
        # self.model.classifier = nn.Identity()

    def forward(self, x):
        x = self.enet(x)
        x = self.metric_classify(self.swish(self.feat(x)))
        
        # x = F.normalize(x, p=2, dim=1)
        # for W in self.fc.parameters():
        #     W = F.normalize(W, p=2, dim=1)
        # x = self.fc(x)
        return x

def build_model(model_name, do_gem):

    model = eff_model(model_name, 512, do_gem=True)
    return model