import torch
from torch import nn
from lavid.models.distill.crd.memory import ContrastMemory
from lavid.models.distill.utils import Embed, Normalize
from lavid.config import TrainConfig

eps = 1e-7


class CRDLoss(nn.Module):
    """
    CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        cfg.s_dim: the dimension of student's feature
        cfg.t_dim: the dimension of teacher's feature
        crd.feat_dim: the dimension of the projection space
        crd.nce_k: number of negatives paired with each positive
        crd.nce_t: the temperature
        crd.nce_m: the momentum for updating the memory buffer
        crd.n_data: the number of samples in the training set, therefor the memory buffer is: crd.n_data x crd.feat_dim
    """
    def __init__(self, cfg: TrainConfig):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(cfg.s_dim, cfg.crd_feat_dim)
        self.embed_t = Embed(cfg.t_dim, cfg.crd_feat_dim)
        self.contrast = ContrastMemory(cfg.crd_feat_dim, cfg.n_data, 4096, 0.07, 0.5)
        self.criterion_t = ContrastLoss(cfg.n_data)
        self.criterion_s = ContrastLoss(cfg.n_data)
        self.feat_dim = cfg.crd_feat_dim

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss
