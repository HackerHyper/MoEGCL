# MoEGCL
## MoEGCL: Mixture of Ego-Graphs Contrastive Representation Learning for Multi-View Clustering
> **Authors:**
Jian Zhu, Xin Zou, Jun Sun, Cheng Luo, Lei Liu, Lingfang Zeng, Ning Zhang, Bian Wu, Chang Tang, Lirong Dai. 

This repo contains the code and data of our AAAI'2026 oral paper.

## 1. Abstract

In recent years, the advancement of Graph Neural Networks (GNNs) has significantly propelled progress in Multi-View Clustering (MVC). However, existing methods face the problem of coarse-grained graph fusion. Specifically, current approaches typically generate a separate graph structure for each view and then perform weighted fusion of graph structures at the view level, which is a relatively rough strategy. To address this limitation, we present a novel Mixture of Ego-Graphs Contrastive Representation Learning (MoEGCL). It mainly consists of two modules. In particular, we propose an innovative Mixture of Ego-Graphs Fusion (MoEGF), which constructs ego graphs and utilizes a Mixture-of-Experts network to implement fine-grained fusion of ego graphs at the sample level, rather than the conventional view-level fusion. Additionally, we present the Ego Graph Contrastive Learning (EGCL) module to align the fused representation with the view-specific representation. The EGCL module enhances the representation similarity of samples from the same cluster, not merely from the same sample, further boosting fine-grained graph representation. Extensive experiments demonstrate that MoEGCL achieves state-of-the-art results in deep multi-view clustering tasks.

## 2.Requirements

pytorch==1.12.1

numpy>=1.21.6

scikit-learn>=1.0.2

## 3.Datasets
LGG


## 4.Usage

- an example for train a new model：

```bash
python main.py
```





If you have any problems, contact me via qijian.zhu@outlook.com.


