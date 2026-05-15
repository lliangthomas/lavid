import torch
import torch.nn as nn
import torch.nn.functional as F

from lavid.models.distill._base import Distiller


def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def rkd_loss(f_s, f_t, squared=False, eps=1e-12, distance_weight=25, angle_weight=50):
    stu = f_s.view(f_s.shape[0], -1)
    tea = f_t.view(f_t.shape[0], -1)

    # RKD distance loss
    with torch.no_grad():
        t_d = _pdist(tea, squared, eps)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td

    d = _pdist(stu, squared, eps)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    loss_d = F.smooth_l1_loss(d, t_d)

    # RKD Angle loss
    with torch.no_grad():
        td = tea.unsqueeze(0) - tea.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = stu.unsqueeze(0) - stu.unsqueeze(1)
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss_a = F.smooth_l1_loss(s_angle, t_angle)

    loss = distance_weight * loss_d + angle_weight * loss_a
    return loss


class RKD(Distiller):
    """Relational Knowledge Disitllation, CVPR2019"""

    def __init__(self, student, teacher):
        super(RKD, self).__init__(student, teacher)
        self.distance_weight = 25
        self.angle_weight = 50
        self.ce_loss_weight = 1
        self.feat_loss_weight = 1
        self.eps = 1e-12
        self.squared = False

    def forward_train(self, image, target, **kwargs):
        feature_student, logits_student = self.student(image, is_feat=True)
        with torch.no_grad():
            feature_teacher, _ = self.teacher(image, is_feat=True)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_rkd = self.feat_loss_weight * rkd_loss(
            feature_student[-1],
            feature_teacher[-1],
            self.squared,
            self.eps,
            self.distance_weight,
            self.angle_weight,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_rkd,
        }
        return logits_student, losses_dict