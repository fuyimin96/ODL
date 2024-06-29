import torch
import torch.nn as nn
import scipy.spatial.distance as spd
import numpy as np

class ODLLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=6, feat_dim=128,margin = 20):
        super(ODLLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.register_buffer("classes", torch.arange(self.num_classes).long())
        self.distcenter = 0
        self.margin = margin
        #self.R = nn.Parameter(torch.zeros((self.num_classes, )))
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        
        self.distcenter = torch.cdist(self.centers, self.centers, p=2)** 2 
        dist_2center,index_2center = torch.sort(self.distcenter)
        nearest_center_dis = dist_2center[:,1]
        nearest_center_index = index_2center[:,1]
        dir_c2c = torch.sub(self.centers.unsqueeze(1).expand(self.num_classes,self.num_classes,self.feat_dim), self.centers)
        #cos_c2c = torch.cosine_similarity(dir_s2c,dir_s2other,dim=2)
        inter_cos = torch.tensor(0.).cuda()
        for i in range(self.num_classes):
            nearest = nearest_center_index[i]
            dir_2nearest = dir_c2c[i,nearest].unsqueeze(0)
            #if nearest_center_index[nearest] != i:
            dir_near2near = dir_c2c[i]
            dis_near2near = self.distcenter[nearest]
            into_compute = (self.classes.ne(i) & self.classes.ne(nearest))
            dis_near2near = torch.masked_select(dis_near2near,into_compute)
            cos_near = torch.cosine_similarity(dir_2nearest,dir_near2near,dim=1)
            cos_near = torch.masked_select(cos_near,into_compute)
            neardis_weight = torch.softmax(-dis_near2near.unsqueeze(0),dim=1)
            incos = ((1+cos_near.unsqueeze(0))*neardis_weight)
            inter_cos += ((1+cos_near.unsqueeze(0))*neardis_weight).sum()
        inter_cos = inter_cos/ self.num_classes
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(self.classes.expand(batch_size, self.num_classes))
        no_mask = labels_expand.ne(self.classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        neighbor_dist = torch.masked_select(distmat,no_mask).view(batch_size,-1)
        dis_weight = torch.softmax(-neighbor_dist,dim=1)

        ### orientation ###
        centers = self.centers.unsqueeze(0).expand(batch_size,self.num_classes,self.feat_dim)
        samples = x.unsqueeze(1).expand(batch_size,self.num_classes,self.feat_dim)
        dir_s2cs = torch.sub(centers, samples)#batch x cls x fea_dim
        dir_s2other = dir_s2cs[no_mask].view(batch_size,-1,self.feat_dim)
        dir_s2c = dir_s2cs[mask]#batch x fea_dim
        dir_s2c = dir_s2c.unsqueeze(1).expand(batch_size,-1,self.feat_dim)
        cosine_dir = torch.cosine_similarity(dir_s2c,dir_s2other,dim=2)
        
        ### repelling ###
        loss_repel = torch.clamp(self.margin - nearest_center_dis, 1e-12, 1e+12).sum()/self.num_classes
        loss_direction = (1-cosine_dir)*dis_weight
        loss_direction = loss_direction.sum() / batch_size
        loss_center = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss=loss_center + loss_direction + 0.1*(loss_repel + inter_cos)
        return loss
