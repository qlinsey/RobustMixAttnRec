from mixtime.models.base import BaseModel

import torch.nn as nn

from abc import *
import torch


class BertBaseModel(BaseModel, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__(args)
        self.ce = nn.CrossEntropyLoss()
        self.noise_reg = args.noise_reg
        self.mse_loss = nn.MSELoss()
        self.tau = 1
        self.sim = "dot"
        self.lmd = args.lmd  # 0.4 #0.2
        self.norm = args.norm
        print("self.lmd=", self.lmd, "self.norm=", self.norm)
        self.batch_size = args.train_batch_size
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.aug_nce_fct = nn.CrossEntropyLoss()

    def forward(self, d):
        logits, info = self.get_logits(d, self.noise_reg)
        # self.noise_reg= True
        # print("d = ",d)
        ret = {"logits": logits, "info": info}
        # print("in BertBaseModel self.noise_reg=",self.noise_reg)
        if self.training:
            labels = d["labels"]
            loss, loss_cnt = self.get_loss(d, logits, labels)
            """
            if self.noise_reg:
                logits1, info1 = self.get_logits(d,self.noise_reg)
                nce_logits, nce_labels = self.info_nce(logits.reshape(logits.shape[0],-1),logits1.reshape(logits1.shape[0],-1), temp=self.tau,batch_size=logits.shape[0], sim=self.sim)
                reg_loss = self.lmd * self.aug_nce_fct(nce_logits, nce_labels)
                #print("reg_loss=",reg_loss) 
            ret['loss'] = loss + reg_loss
            ret['loss'] = loss
            """
            ##for no_noise
            if self.noise_reg:
                if self.norm:
                    ret["loss"] = loss + self.lmd * info["xnorm"] * self.mse_loss(
                        info["org"], info["noise"]
                    )
                else:
                    ret["loss"] = loss + self.lmd * self.mse_loss(
                        info["org"], info["noise"]
                    )
            else:
                ret["loss"] = loss
            ret["loss_cnt"] = loss_cnt
        else:
            # get scores (B x V) for validation
            last_logits = logits[:, -1, :]  # B x H
            ret["scores"] = self.get_scores(d, last_logits)  # B x C
        return ret

    @abstractmethod
    def get_logits(self, d):
        pass

    @abstractmethod
    def get_scores(self, d, logits):  # logits : B x H or M x H, returns B x C or M x V
        pass

    def get_loss(self, d, logits, labels):
        _logits = logits.view(-1, logits.size(-1))  # BT x H
        _labels = labels.view(-1)  # BT

        valid = _labels > 0
        loss_cnt = valid.sum()  # = M
        valid_index = valid.nonzero().squeeze()  # M

        valid_logits = _logits[valid_index]  # M x H
        valid_scores = self.get_scores(d, valid_logits)  # M x V
        valid_labels = _labels[valid_index]  # M

        loss = self.ce(valid_scores, valid_labels)
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt

    def info_nce(self, z_i, z_j, temp, batch_size, sim="dot"):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        if sim == "cos":
            sim = (
                nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
                / temp
            )
        elif sim == "dot":
            sim = torch.matmul(z, z.T) / temp
            # print("sim=",sim)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
