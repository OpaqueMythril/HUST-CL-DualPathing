from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function


class PromptGenerator(nn.Module):
    def __init__(self, embed_dim=768, prompt_len=8):
        super(PromptGenerator, self).__init__()
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        # 将 768 维特征映射为提示向量
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim * prompt_len)
        )

    def forward(self, x):
        # x: [batch, 768] -> [batch, Lp, 768]
        out = self.net(x)
        return out.view(-1, self.prompt_len, self.embed_dim)


class Prompt(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):
        with torch.no_grad():
            m = self.model.module if hasattr(self.model, 'module') else self.model
            backbone = m.feat if hasattr(m, 'feat') else m  # 在 zoo.py 中 ViT 被命名为 feat

            #调用vit.py 中forward_features
            if hasattr(backbone, 'forward_features'):
                # forward_features 通常返回包含所有 token 的 tensor [Batch, Tokens, 768]
                raw_features = backbone.forward_features(inputs)
            else:
                out = backbone(inputs, train=False)
                raw_features = out[0] if isinstance(out, (tuple, list)) else out

            # 提取 [CLS] token: [Batch, Tokens, 768] -> [Batch, 768]
            features = raw_features[:, 0, :] if raw_features.dim() == 3 else raw_features

            # 维度硬核检查
            if features.shape[-1] != 768:
                raise RuntimeError(f"特征提取维度错误：得到 {features.shape[-1]}，需要 768。")

        # 2. 向量空间分解逻辑：v = v_proj + v_diff
        v_proj, v_diff = self.decompose_features(features)

        # 3. 双路提示生成与门控融合
        p_shared = self.g_shared(v_proj)
        p_specific = self.g_specifics[self.task_id](v_diff)

        # 门控融合
        g = self.gate(features)
        p_final = g.unsqueeze(-1) * p_shared + (1 - g).unsqueeze(-1) * p_specific

        # 4. 正式训练前向传播：注入生成的提示
        p_list = [p_final, p_final]
        logits, prompt_loss = self.model(inputs, prompt=p_list, train=True)

        # 处理 Logits 并映射
        if logits.dim() == 3: logits = logits[:, 0, :]

        # 映射至当前有效分类空间
        logits = logits[:, :self.valid_out_dim]
        logits[:, :self.last_valid_out_dim] = -float('inf')

        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls) + prompt_loss.sum()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    def decompose_features(self, v_new):
        """实现方案中的几何分解"""
        if len(self.knowledge_base) == 0:
            return v_new, torch.zeros_like(v_new).to(v_new.device)
        all_protos = torch.stack([p for p in self.knowledge_base.values()]).to(v_new.device)
        sim = torch.matmul(F.normalize(v_new, dim=-1), F.normalize(all_protos, dim=-1).t())
        _, topk_idx = torch.topk(sim, k=min(5, len(all_protos)), dim=-1)
        v_base = all_protos[topk_idx].mean(dim=1)
        v_proj = (torch.sum(v_new * v_base, dim=-1, keepdim=True) / (
                    torch.sum(v_base * v_base, dim=-1, keepdim=True) + 1e-8)) * v_base
        v_diff = v_new - v_proj
        return v_proj, v_diff

    def init_optimizer(self):
        # 包含新增组件的参数
        params_to_opt = list(self.g_shared.parameters()) + \
                        list(self.g_specifics.parameters()) + \
                        list(self.gate.parameters())

        curr_model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(curr_model, 'last'):
            params_to_opt += list(curr_model.last.parameters())

        optimizer_arg = {'params': params_to_opt, 'lr': self.config['lr'], 'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'], 0.999)
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        self.g_shared = self.g_shared.cuda()
        self.g_specifics = self.g_specifics.cuda()
        self.gate = self.gate.cuda()
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'],
                                               output_device=self.config['gpuid'][0])
        return self


class CODAPrompt(Prompt):
    def __init__(self, learner_config):
        # 强制手动初始化
        nn.Module.__init__(self)
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        # 属性定义
        self.num_tasks = self.prompt_param[0]
        self.g_shared = PromptGenerator()
        self.g_specifics = nn.ModuleList([PromptGenerator() for _ in range(self.num_tasks)])
        self.gate = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
        self.knowledge_base = {}
        self.task_id = 0

        # 模型创建
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](
            out_dim=self.out_dim, prompt_flag='coda', prompt_param=self.prompt_param)

        for name, param in model.named_parameters():
            if "last" not in name: param.requires_grad = False
        return model

    def process_task_count(self):
        self.g_specifics[self.task_id].eval()
        for param in self.g_specifics[self.task_id].parameters():
            param.requires_grad = False
        self.task_id += 1


class DualPrompt(Prompt):
    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        return models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag='dual',
                                                                              prompt_param=self.prompt_param)


class L2P(Prompt):
    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        return models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag='l2p',
                                                                              prompt_param=self.prompt_param)