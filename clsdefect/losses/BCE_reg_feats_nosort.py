import torch.nn as nn
import torch


class Loss(nn.Module):
    def __init__(self, params="", **kwargs):
        super(Loss, self).__init__()
        w = params.split("_")
        assert len(w) == 4
        self.losses = nn.ModuleList([nn.BCELoss()]*3)
        self.weights = [float(wi) for wi in w[:3]]
        self.feats_sim = nn.CosineSimilarity()
        self.feats_w = float(w[3])
        assert self.feats_w >= 0.

    def forward(self, model_output, masks, target):
        out = sum([w*loss(model_output['pmf'][:, i], target[:, i]) for i, (w, loss) in enumerate(zip(self.weights, self.losses))])
        return out + self.feats_w*self.feats_sim(model_output['features']['general'], model_output['features']['delam']).mean()
