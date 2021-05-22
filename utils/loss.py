import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def focal_loss(inputs, targets, alpha=1, gamma=0, size_average=True, ignore_index=255):
    ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    if size_average:
        return focal_loss.mean()
    else:
        return focal_loss.sum()

def kldiv(logits, targets, reduction='batchmean'):
    p = F.log_softmax(logits, dim=1)
    q = F.softmax(targets, dim=1)
    return F.kl_div(p, q, reduction=reduction)


class InstanceSimilarity(nn.Module):
    '''
    Instance Similarity based loss
    '''
    def __init__(self, mse=True):
        super(InstanceSimilarity, self).__init__()
        self.mse = mse

    def _loss(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s  = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t  = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t) if self.mse else F.l1_loss(norm_G_s, norm_G_t)
        return loss

    def forward(self, g_s, g_t):
        return sum(self._loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t))



class SCRM(nn.Module):
    """
    spatial & channel wise relation loss
    """
    def __init__(self, gamma=0.1):
        super(SCRM, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = gamma

    def spatial_wise(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = x.view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

    def channel_wise(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

    def cal_loss(self, f_s, f_t):
        f_s = F.normalize(f_s, dim=1)
        f_t = F.normalize(f_t, dim=1)
        sa_loss = F.l1_loss(self.spatial_wise(f_s), self.spatial_wise(f_t))
        ca_loss = F.l1_loss(self.channel_wise(f_s), self.channel_wise(f_t))
        return ca_loss + sa_loss

    def forward(self, g_s, g_t):
        return sum(self.cal_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t))

