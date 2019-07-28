
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# # Total Loss Function

# In[51]:


class TotalLoss(nn.modules.loss._Loss):
    def __init__(self, loss_funcs):
        super(TotalLoss, self).__init__()
        self.loss_funcs = loss_funcs
        self.ite = 1

    def forward(self, model_id, outputs, label_id, log=None, **kwargs):
        losses = []
        target_output = outputs[model_id]
        for source_id, (source_output, loss_func) in enumerate(zip(outputs, self.loss_funcs)):
            kwargs["_source_id"] = source_id
            kwargs["_target_id"] = model_id
            losses += [loss_func(target_output,
                                 source_output,
                                 label_id, 
                                 log,
                                 **kwargs)]
        total_loss = torch.stack(losses).sum()
        
        if log is not None:
            for source_id, (func, loss) in enumerate(zip(self.loss_funcs, losses)):
                log["ite_log"][self.ite][f"{model_id:02}_{source_id:02}_"+func.__class__.__name__] = float(loss)
            log["ite_log"][self.ite][f"{model_id:02}_loss"] = float(total_loss)
            self.ite += 1
        return total_loss


# # Loss Functioin


# ## Base Class

# In[7]:


class _LossBase(nn.modules.loss._Loss):
    def __init__(self, args):
        super(_LossBase, self).__init__()
        self.args = args
        self.ite = 1

    def forward(self, target_output, source_output, label_id, log=None, **kwargs):
        return 


# ## Independent Loss

# In[9]:


class IndependentLoss(_LossBase):
    def __init__(self, args):
        super(IndependentLoss, self).__init__(args)
        self.gate = globals()[args.gate.name](self, args.gate.args)
        return

    def forward(self, target_output, source_output, label_id, log=None, **kwargs):
        loss_per_sample = F.cross_entropy(target_output, label_id, reduction='none')
        
        kwargs["_student_logits"] = target_output        
        identity = torch.eye(target_output.shape[-1], device=target_output.device)
        onehot_label = identity[label_id]
        kwargs["_teacher_logits"] = onehot_label
        kwargs["_label_id"] = label_id
        
        hard_loss = self.gate.f(loss_per_sample, log, **kwargs)        
        
        loss = hard_loss * self.args.loss_weight
        
        if log is not None:
            self.ite = self.ite + 1
            
        return loss


# ## KL Distillation Loss

# In[73]:


class KLLoss(_LossBase):
    def __init__(self, args):
        super(KLLoss, self).__init__(args)
        self.T = args.T
        self.gate = globals()[args.gate.name](self, args.gate.args)
        return

    def forward(self, target_output, source_output, label_id, log=None, **kwargs):        
        student_logits = target_output
        teacher_logits = source_output.detach()
                
        student_softmax = F.softmax(student_logits/self.T, dim=1)
        teacher_softmax = F.softmax(teacher_logits/self.T, dim=1)
        
        soft_loss_per_sample = self.kl_divergence(student_softmax, teacher_softmax, log=log)
        
        kwargs["_student_logits"] = student_logits
        kwargs["_teacher_logits"] = teacher_logits
        kwargs["_label_id"] = label_id
        
        soft_loss = self.gate.f(soft_loss_per_sample, log, **kwargs)
                
        loss = soft_loss * (self.T**2)
                         
        if log is not None:
            self.ite = self.ite + 1
        
        return loss
    
    def kl_divergence(self, student, teacher, log=None):
        kl = teacher * torch.log((teacher / (student+1e-10)) + 1e-10)
        kl = kl.sum(dim=1)        
        loss = kl
        return loss


# # Gate

# ## Base

# In[26]:


class _BaseGate():
    def __init__(self, parent, args):
        self.parent = parent
        self.args = args
        
    def f(self, loss_per_sample, log, **kwargs):
        return soft_loss


# ## Through

# In[ ]:


class ThroughGate(_BaseGate):
    def f(self, loss_per_sample, log, **kwargs):
        soft_loss = loss_per_sample.mean()
        return soft_loss


# ## Cutoff

# In[ ]:


class CutoffGate(_BaseGate):
    def f(self, loss_per_sample, log, **kwargs):
        soft_loss = torch.zeros_like(loss_per_sample[0], requires_grad=True).sum()            
        return soft_loss


# ## Linear

# In[1]:


class LinearGate(_BaseGate):
    def f(self, loss_per_sample, log, **kwargs):
        if log is not None:
            self.end_ite = log.iteration
            
        loss_weight = self.parent.ite / self.end_ite
        
        soft_loss = loss_per_sample.mean()
        soft_loss = soft_loss * loss_weight
                
        if log is not None:
            source_id = kwargs["_source_id"]
            target_id = kwargs["_target_id"]
            log["ite_log"][self.parent.ite][f"{target_id:02}_{source_id:02}_linear_weight"] = float(loss_weight)
            
        return soft_loss


# ## Correct

# In[1]:


class CorrectGate(_BaseGate):    
    def f(self, loss_per_sample, log, **kwargs):
        student_logits = kwargs["_student_logits"]
        teacher_logits = kwargs["_teacher_logits"]
        label_id = kwargs["_label_id"]
        
        true_s = student_logits.argmax(dim=1) == label_id
        true_t = teacher_logits.argmax(dim=1) == label_id
        TT = ((true_t == 1) & (true_s == 1)).type_as(loss_per_sample[0])
        TF = ((true_t == 1) & (true_s == 0)).type_as(loss_per_sample[0])
        FT = ((true_t == 0) & (true_s == 1)).type_as(loss_per_sample[0])
        FF = ((true_t == 0) & (true_s == 0)).type_as(loss_per_sample[0])
        mask = 1*TT + 1*TF + 0*FT + 0*FF
        
        soft_loss = (loss_per_sample * mask).mean()
                
        if log is not None:
            source_id = kwargs["_source_id"]
            target_id = kwargs["_target_id"]
            log["ite_log"][self.parent.ite][f"{target_id:02}_{source_id:02}_TT"] = float(TT.sum())
            log["ite_log"][self.parent.ite][f"{target_id:02}_{source_id:02}_TF"] = float(TF.sum())
            log["ite_log"][self.parent.ite][f"{target_id:02}_{source_id:02}_FT"] = float(FT.sum())
            log["ite_log"][self.parent.ite][f"{target_id:02}_{source_id:02}_FF"] = float(FF.sum())
        
        return soft_loss

