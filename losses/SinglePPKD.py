import torch
import torch.nn as nn
import torch.nn.functional as F
class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """
    def __init__(self, temperature, alpha=None, beta=None, p=0.5,reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.p=p
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        b1_indices = torch.arange(targets.shape[0]) % 2 == 0
        b2_indices = torch.arange(targets.shape[0]) % 2 != 0

        original_soft_loss = super().forward(torch.log_softmax(student_output[b1_indices] / self.temperature, dim=1),
                                    torch.softmax(teacher_output[b1_indices] / self.temperature, dim=1))
        b1_max=torch.gather(teacher_output[b1_indices],1,targets[b1_indices].unsqueeze(-1))
        b2_max=torch.gather(teacher_output[b2_indices],1,targets[b2_indices].unsqueeze(-1))
        augment_student_output=torch.where(torch.abs(b1_max-b2_max)>self.p,student_output[b2_indices].clone().detach(),student_output[b2_indices])
        augmented_soft_loss = super().forward(torch.log_softmax(augment_student_output / self.temperature, dim=1),
                                    torch.softmax(augment_student_output / self.temperature, dim=1))
        soft_loss=(original_soft_loss+augmented_soft_loss)/2
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss
        hard_loss = self.cross_entropy_loss(student_output, targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss
