import torch
import torch.nn as nn
import torch.nn.functional as F

class MaKD(nn.Module):
    def __init__(self, num_classes):
        super(MaKD, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, teacher):
        student = x[:, self.num_classes:]
        
        B = teacher.shape[0]
        teacher = teacher.reshape(B, -1)
        loss = F.binary_cross_entropy_with_logits(student, teacher, reduction='mean')

        return loss