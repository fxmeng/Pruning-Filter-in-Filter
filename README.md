# Pruning-Filter-in-Filter

This is the PyTorch implementation of our NeurIPS 2020 paper "Pruning Filter in Filter". In this paper:
1) We propose a new pruning paradigm called Stripe-wise-Pruning(SP), which can be seen as a general case of the Filter-Pruning (FP). SP treats a filter $F \in \mathbb{R}^{C\times K\times K}$as $K\times K$ stripes (i.e., $1\times 1$ filters $\in \mathbb{R}^c $ ), and perform pruning at the unit of stripe instead of the whole filter. Compared to the existing methods, SP achieves finer granularity than traditional FP while being hardware friendly than Weight-Pruning and keeping independence between filters than Group-wise-Pruning, leading to state-of-the-art pruning ratio on CIFAR-10 and ImageNet. 

2) More arousingly, by applying SP, we find another important property of filters alongside their weight: shape. Start with a random initialized ResNet56, we train and trim the shape of filters and boost the test accuracy from 10.00% to 80.58% on CIFAR-10 without updating the filter weights. The optimal shape of filters are learned by the proposed Filter-Skeleton (FS) in the paper and we believe FS could inspire further research towards the essence of network pruning.

**Related code will be coming soon...**
![image](BrokenNet_filter.png)
