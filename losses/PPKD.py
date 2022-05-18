import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

class ProPadKD(nn.Module):
    def __init__(self,temperature,step,num_classes,alpha=1, beta=1, reduction='batchmean', **kwargs):
        super(ProPadKD, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.step=step
        self.num_classes=num_classes
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)
        self.kl_loss=nn.KLDivLoss(reduction=reduction)
        self.buffer = [list() for i in range(self.num_classes)]
        number=int(1/self.step)+1
        self.x=np.linspace(0,1,number)
    def guasscdf(self,x,u,o):
        result=[]
        for i in x:
            result.append(stats.norm.cdf(i,u,o))
        for i in range(len(result)-1):
            result[i]=result[i+1]-result[i]
        result=[i/sum(result[:-1]) for i in result[:-1]]
        return result
    def forward(self, student_output, teacher_output, targets=None, buffer=False,*args, **kwargs):
        """===================================================Choose====================================================="""
        if buffer:
            b2_indices = torch.arange(targets.shape[0]) % 2 == 0
            buffer_logits = teacher_output[b2_indices]
            buffer_target = targets[b2_indices]
            tlogits = torch.gather(torch.softmax(buffer_logits, 1), 1, buffer_target.unsqueeze(-1)).tolist()
            ttarget = buffer_target.tolist()
            for t,l in zip(ttarget,tlogits):
                self.buffer[t].append(l)
        else:
            if not hasattr(self.y):
                self.y=[]
                for i, bf in enumerate(self.buffer):
                    self.buffer[i] = np.array(self.buffer[i])
                    mean, std = self.buffer[i].mean(), self.buffer[i].std()
                    self.y.append(self.guasscdf(self.x, mean, std))
                    # y->(num_classes,1/step)
                """===============================================vanilla KD Loss================================================="""
                soft_loss = self.kl_loss(torch.log_softmax(student_output / self.temperature, dim=1),
                                            torch.softmax(teacher_output / self.temperature, dim=1))
                if self.alpha is None or self.alpha == 0 or targets is None:
                    return soft_loss
                hard_loss = self.cross_entropy_loss(student_output, targets)
                KDLoss=self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss
                return KDLoss
            else:
                b1_indices = torch.arange(targets.shape[0]) % 2 == 0
                b2_indices = torch.arange(targets.shape[0]) % 2 != 0
                student_logits=torch.softmax(student_output[b2_indices],1)
                
                origin_soft_loss = self.kl_loss(torch.log_softmax(student_output[b1_indices] / self.temperature, dim=1),
                                            torch.softmax(teacher_output[b1_indices] / self.temperature, dim=1))
                augment_soft_loss = self.kl_loss(torch.log_softmax(student_output[b2_indices] / self.temperature, dim=1),
                                            torch.softmax(teacher_output[b2_indices] / self.temperature, dim=1))
                soft_loss=(origin_soft_loss+augment_soft_loss)/2
                if self.alpha is None or self.alpha == 0 or targets is None:
                    return soft_loss
                hard_loss = self.cross_entropy_loss(student_output, targets)
                KDLoss=self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss
                return KDLoss

