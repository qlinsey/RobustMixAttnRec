from .bert_base import BertBaseModel
from .embeddings import *
from .bodies import MixtimeBody
from .heads import *

import torch
import torch.nn as nn


class MixtimeModel(BertBaseModel):
    def __init__(self, args):
        super().__init__(args)
        hidden = args.hidden_units
        self.output_info = args.output_info
        absolute_kernel_types = args.absolute_kernel_types
        relative_kernel_types = args.relative_kernel_types
        ##### Footers
        # Token Embeddings
        self.token_embedding = TokenEmbedding(args)
        # Absolute Embeddings
        self.absolute_kernel_embeddings_list = nn.ModuleList()
        if absolute_kernel_types is not None and len(absolute_kernel_types) > 0:
            for kernel_type in absolute_kernel_types.split("-"):
                if kernel_type == "p":  # position
                    emb = PositionalEmbedding(args)
                elif kernel_type == "d":  # day
                    emb = DayEmbedding(args)
                elif kernel_type == "c":  # constant
                    emb = ConstantEmbedding(args)
                elif kernel_type == "b":
                    # (self, d: int, g: int, n_attrs: int, enc: Encoding)
                    emb = BochnerEmbedding(args)
                    # emb = AttrEmbedding(args) #args.hidden_units,n_out,args.num_items)
                else:
                    raise ValueError
                self.absolute_kernel_embeddings_list.append(emb)
        # Relative Embeddings
        self.relative_kernel_embeddings_list = nn.ModuleList()
        if relative_kernel_types is not None and len(relative_kernel_types) > 0:
            for kernel_type in relative_kernel_types.split("-"):
                if kernel_type == "s":  # time difference
                    emb = SinusoidTimeDiffEmbedding(args)
                elif kernel_type == "e":
                    emb = ExponentialTimeDiffEmbedding(args)
                elif kernel_type == "l":
                    emb = Log1pTimeDiffEmbedding(args)
                else:
                    raise ValueError
                self.relative_kernel_embeddings_list.append(emb)
        # Lengths
        self.La = len(self.absolute_kernel_embeddings_list)
        self.Lr = len(self.relative_kernel_embeddings_list)
        self.L = self.La + self.Lr
        # Sanity check
        assert hidden % self.L == 0, "multi-head has to be possible"
        assert (
            len(self.absolute_kernel_embeddings_list) > 0
            or len(self.relative_kernel_embeddings_list) > 0
        )
        ##### BODY
        self.body = MixtimeBody(args, self.La, self.Lr)
        ##### Heads
        # self.bert_head = BertDotProductPredictionHead(args)
        if args.headtype == "dot":
            self.head = BertDotProductPredictionHead(args, self.token_embedding.emb)
        elif args.headtype == "linear":
            self.head = BertLinearPredictionHead(args)
        else:
            raise ValueError
        ##### dropout
        self.dropout = nn.Dropout(p=args.dropout)
        ##### Weight Initialization
        self.init_weights()
        ##### MISC
        self.ce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    @classmethod
    def code(cls):
        return "mixtime"

    def get_logits(self, d, noise_reg):
        x = d["tokens"]
        e_mask = torch.where(x == 0.0, 0.0, 1.0)
        attn_mask = (
            (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        )  # B x 1 x T x T
        token_embeddings = self.dropout(self.token_embedding(d))  # B x T x H

        # token_embeddings = token_embeddings.unsqueeze(0).expand(self.L, -1, -1, -1)  # L x B x T x H
        # token_embeddings = token_embeddings.chunk(self.L, dim=0)  # L of [1 x B x T x H]
        # token_embeddings = [x.squeeze(0) for x in token_embeddings]  # L of [B x T x H]

        # absolute_kernel_embeddings = [self.dropout(emb(d, e_mask)) for emb in self.absolute_kernel_embeddings_list]  # La of [B x T x H]
        absolute_kernel_embeddings = [
            self.dropout(emb(d)) for emb in self.absolute_kernel_embeddings_list
        ]
        relative_kernel_embeddings = [
            self.dropout(emb(d)) for emb in self.relative_kernel_embeddings_list
        ]  # Lr of [B x T x T x H]

        info = {} if self.output_info else None

        # last_hidden = L of [B x T x H]
        last_hidden = self.body(
            False,
            token_embeddings,
            attn_mask,
            absolute_kernel_embeddings,
            relative_kernel_embeddings,
            info=info,
        )
        # last_hidden2 = self.body(True,token_embeddings, attn_mask,absolute_kernel_embeddings,relative_kernel_embeddings,info=info)
        # last_hidden = torch.cat(last_hidden, dim=-1)  # B x T x LH
        info = {}
        if noise_reg:
            print(" in noise_reg info")
            last_hidden2 = self.body(
                True,
                token_embeddings,
                attn_mask,
                absolute_kernel_embeddings,
                relative_kernel_embeddings,
                info=info,
            )
            info["org"] = last_hidden
            info["noise"] = last_hidden2
            info["xnorm"] = torch.linalg.norm(token_embeddings, dim=1, ord=2)
        return last_hidden, info

    def get_scores(self, d, logits):
        # logits : B x H or M x H
        if self.training:  # logits : M x H, returns M x V
            h = self.head(logits)  # M x V
        else:  # logits : B x H,  returns B x C
            candidates = d["candidates"]  # B x C
            h = self.head(logits, candidates)  # B x C
        return h
