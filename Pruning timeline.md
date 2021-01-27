













#               Awesome Pruning

​       









                                                                            ##                                                                           ---By Rumi Meng(孟繁续)





















| Title | [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) |
|:-------|:---------|
| **Authors** | [Song Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+S), [Jeff Pool](https://arxiv.org/search/cs?searchtype=author&query=Pool%2C+J), [John Tran](https://arxiv.org/search/cs?searchtype=author&query=Tran%2C+J), [William J. Dally](https://arxiv.org/search/cs?searchtype=author&query=Dally%2C+W+J) |
| **Venue** | NeurIPS 2015 |
| **Model&Dataset&Code** | LeNet on MNIST,AlexNet and VGG-16 on ImageNet,[PyTorch(3rd)](https://github.com/jack-willturner/DeepCompression-PyTorch) |
| **Type** | `W` |

Neural networks are both computationally intensive and memory intensive, making them difficult to deploy on embedded systems. Also, conventional networks fix the architecture before training starts; as a result, training cannot improve the architecture. <br />3 Learning Connections in Addition to Weights<br />3.1 Regularization---L2 norm is better than L1 norm<br />3.2 Dropout Ratio Adjustment <br />
$$
C_i = N_i N_{i-1}~~~~~~~~~~~~~~
D_r = D_o \sqrt{\frac{C_{ir}}{C_{io}}}
$$
3.3 Local Pruning and Parameter Co-adaptation---keep the surviving parameters instead of re-initializing them.Fix CONV layers and only retrain the FC layers<br />3.4 Iterative Pruning---each iteration is a greedy search <br />3.5 Pruning Neurons---neurons with zero input connections or zero output connections may be safely prune

![network_simple](./fig/network_simple.png)



| Title                  | [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Song Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+S), [Huizi Mao](https://arxiv.org/search/cs?searchtype=author&query=Mao%2C+H), [William J. Dally](https://arxiv.org/search/cs?searchtype=author&query=Dally%2C+W+J) |
| **Venue**              | ICLR 2016 **(Best)**                                         |
| **Model&Dataset&Code** | LeNet on MNIST,AlexNet and VGG-16 on ImageNet,[Caffe(Author)](https://github.com/songhan/Deep-Compression-AlexNet) |
| **Type**               | `W`                                                          |



Use pruning, trained quantization and Huffman coding to reduce the storage and energy required to run inference. For pruned AlexNet, we are able to quantize to 8-bits (256 shared weights) for each CONV layers, and 5-bits (32 shared weights) for each FC layer without any loss of accuracy.We use k-means clustering to identify the shared weights for each layer of a trained network.

![quantization_hori](./fig/quantization_hori.png)



![jump](./fig/jump.png)

using compressed sparse row (CSR) or compressed sparse column (CSC) format, which requires 2a + n + 1 numbers, where a is the number of non-zero elements and n is the number of rows or columns.

![theory](./fig/theory.png)

| Method                    | Library                              | Type | Hardware                                         |
| ------------------------- | ------------------------------------ | ---- | ------------------------------------------------ |
| benchmark<br />CSR format | cuBLAS GEMV<br />cuSPARSE CSRMV      | GPU  | NVIDIA GeForce GTX Titan X<br />&NVIDIA Tegra K1 |
| benchmark<br />CSR format | MKL CBLAS GEMV<br />MKL SPBLAS CSRMV | CPU  | Intel Core i7 5930K                              |

While the *pruned* network has been benchmarked on various hardware, the *quantized* network with weight sharing has not, because off-the-shelf cuSPARSE or MKL SPBLAS library does not support indirect matrix entry lookup, nor is the relative index in CSC or CSR format supported. 





| Title                  | [High performance convolutional neural networks for document processing](https://hal.inria.fr/inria-00112631/) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [K Chellapilla](https://scholar.google.com/citations?user=R5RXd_0AAAAJ&hl=zh-CN&oi=sra), S Puri, [P Simard](https://scholar.google.com/citations?user=akrkYU0AAAAJ&hl=zh-CN&oi=sra)@[Microsoft Research](https://hal.inria.fr/search/index/q/*/structId_i/23526/) |
| **Venue**              | *Tenth International Workshop on Fron- tiers in Handwriting Recognition*, 2006. |
| **Model&Dataset&Code** |                                                              |
| **Type**               |                                                              |

![Unrolling the convolution](fig/Unrolling the convolution.png)

| Title                  | [Channel-level acceleration of deep face representations](https://www.infona.pl/resource/bwmeta1.element.ieee-art-000007303876) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [A Polyak](https://scholar.google.com/citations?user=CP62OTMAAAAJ&hl=zh-CN&oi=sra), [L Wolf](https://scholar.google.com/citations?user=UbFrXTsAAAAJ&hl=zh-CN&oi=sra) |
| **Venue**              | IEEE Access, 2015                                            |
| **Model&Dataset&Code** |                                                              |
| **Type**               | `Inbound Pruning`                                            |

![Inbound](fig/Inbound.png)





| Title                  | [Dynamic Network Surgery for Efficient DNNs](https://arxiv.org/abs/1608.04493) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Yiwen Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+Y), [Anbang Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+A), [Yurong Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y) |
| **Venue**              | NeurIPS 2016                                                 |
| **Model&Dataset&Code** | LeNet on MNIST,AlexNet and VGG-16 on ImageNet,[Caffe(Author)](https://github.com/yiwenguo/Dynamic-Network-Surgery) |
| **Type**               | `W`                                                          |

That is, a network connection may be redundant due to the existence of some others, but it will soon become crucial once the others are removed. 
$$
\min_{\mathbf W_k,\mathbf T_k} L\left(\mathbf W_k\odot \mathbf T_k\right) \quad \mathrm{s.t.} \ \mathbf T^{(i,j)}_k = \mathbf h_k(\mathbf W^{(i,j)}_k),\ \forall (i,j) \in \mathcal I,\\
\mathbf W^{(i,j)}_k \leftarrow \mathbf W^{(i,j)}_k - \beta \frac{\partial}{\partial (\mathbf W^{(i,j)}_k \mathbf T^{(i,j)}_k )}L\left(\mathbf W_k\odot \mathbf T_k \right) ,\ \forall (i,j) \in \mathcal I,\\
\mathbf h_k (\mathbf W^{(i,j)}_k)=\left\{ \begin{array}{ll}
0&\text{if }a_k>|\mathbf W^{(i,j)}_k| \\
\mathbf T^{(i,j)}_k &\text{if }a_k\leq|\mathbf W^{(i,j)}_k|<b_k \\
1&\text{if } b_k\leq |\mathbf W^{(i,j)}_k| \\
\end{array} \right.
$$
![1a2](./fig/1a2.png)





| Title                  | [Perforatedcnns: Acceleration through elimination of redundant convolutions](https://arxiv.org/abs/1504.08362) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Michael Figurnov](https://arxiv.org/search/cs?searchtype=author&query=Figurnov%2C+M), [Aijan Ibraimova](https://arxiv.org/search/cs?searchtype=author&query=Ibraimova%2C+A), [Dmitry Vetrov](https://arxiv.org/search/cs?searchtype=author&query=Vetrov%2C+D), [Pushmeet Kohli](https://arxiv.org/search/cs?searchtype=author&query=Kohli%2C+P) |
| **Venue**              | NIPS 2016                                                    |
| **Model&Dataset&Code** | NIN on CIFAR-10, AlexNet and VGG-16 on ImageNet              |
| **Type**               | `Spatial`                                                    |

![im2row-crop](fig/im2row-crop.png)



| Title                  | [Fast ConvNets Using Group-wise Brain Damage](https://arxiv.org/abs/1506.02515) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Vadim Lebedev](https://arxiv.org/search/cs?searchtype=author&query=Lebedev%2C+V), [Victor Lempitsky](https://arxiv.org/search/cs?searchtype=author&query=Lempitsky%2C+V) |
| **Venue**              | CVPR 2016                                                    |
| **Model&Dataset&Code** | LeNet on MNIST，AlexNet and VGG19 on ImageNet                |
| **Type**               | `Group`                                                      |

Group lasso:
$$
Omega_{2,1}(K) = \lambda \sum_{i,j,s} \|\Gamma_{ijs}\| = \lambda \sum_{i,j,s} \sqrt{ \sum_{t=1}^T K(i,j,s,t)^2 }\\
\frac{ \partial \Omega_{2,1}(K)} {\partial K(i,j,s,t)}  = \lambda \frac{K(i,j,s,t)}{ \sqrt{ \sum_{z=1}^{T} K(i,j,s,z)^2 } }
$$
![groupSparse](fig/groupSparse.png)

| Title                  | [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Hao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Asim Kadav](https://arxiv.org/search/cs?searchtype=author&query=Kadav%2C+A), [Igor Durdanovic](https://arxiv.org/search/cs?searchtype=author&query=Durdanovic%2C+I), [Hanan Samet](https://arxiv.org/search/cs?searchtype=author&query=Samet%2C+H), [Hans Peter Graf](https://arxiv.org/search/cs?searchtype=author&query=Graf%2C+H+P) |
| **Venue**              | ICLR 2017                                                    |
| **Model&Dataset&Code** | VGG-16,ResNet-56/110 on CIFAR-10, ResNet-34 on ImageNet,[PyTorch(3rd)](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet/l1-norm-pruning) |
| **Type**               | `F`                                                          |

By removing whole filters in the network together with their connecting feature maps, the computation costs are reduced significantly. In contrast to pruning weights, this approach does not result in sparse connectivity patterns. Hence, it does not need the support of sparse convolution libraries and can work with existing efficient BLAS libraries for dense matrix multiplications. 

![pruning1](./fig/pruning1.png)



Calculate the sum of each filters' L1 norm and sort by it. Prune m filters  with the smallest sum values and their corresponding feature maps and the kernels in the next convolutional layer corresponding to the pruned feature maps.

we prune each layer independently and evaluate the resulting pruned network’s accuracy on the validation set.On the contrary, layers with relatively flat slopes are more sensitive to pruning. We empirically determine the number of filters to prune for each layer based on their sensitivity to pruning. we observe that layers in the same stage (with the same feature map size) have a similar sensitivity to pruning. To avoid introducing layer-wise meta-parameters, we use the same pruning ratio for all layers in the same stage. For layers that are sensitive to pruning, we prune a smaller percentage of these layers or completely skip pruning them.

![prune_every_layer_by_l1_norm](fig/prune_every_layer_by_l1_norm.png)

![greedy](fig/greedy.png)

The **independent** pruning strategy calculates the filter sum (columns marked in green) without considering feature maps removed in previous layer (shown in blue), so the kernel weights marked in yellow are still included. The **greedy** pruning strategy does not count kernels for the already pruned feature maps. 

![shortcut](fig/shortcut.png)

Since the identical feature maps are more important than the added residual maps, the feature maps to be pruned should be determined by the pruning results of the shortcut layer. 

VGG and Residual networks have fewer parameters in the fully connected layers. Hence, pruning a large percentage of parameters from these networks is challenging.

| Title                  | [Channel pruning for accelerating very deep neural networks](https://arxiv.org/abs/1707.06168) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Yihui He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+Y), [Xiangyu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X), [Jian Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+J) |
| **Venue**              | ICCV 2017                                                    |
| **Model&Dataset&Code** |                                                              |
| **Type**               | `C`                                                          |

![ill](fig/ill.png)





| Title                  | [Runtime Neural Pruning](https://papers.neurips.cc/paper/6813-runtime-neural-pruning) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | *Ji Lin, Yongming Rao, Jiwen Lu, Jie Zhou*                   |
| **Venue**              | NeurIPS 2017                                                 |
| **Model&Dataset&Code** |                                                              |
| **Type**               | `W`                                                          |

![Runtime Neural Pruning](/Users/rumimeng/Documents/paper/fig/Runtime Neural Pruning.png)



| Title                  | [Learning Efficient Convolutional Networks Through Network Slimming](https://arxiv.org/abs/1708.06519) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Zhuang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Jianguo Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Zhiqiang Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+Z), [Gao Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+G), [Shoumeng Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+S), [Changshui Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+C) |
| **Venue**              | ICCV 2017                                                    |
| **Model&Dataset&Code** | [PyTorch(Author)](https://github.com/Eric-mingjie/network-slimming) |
| **Type**               | `F`                                                          |

$$
\hat{z} = \frac{z_{in} - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}; \ \
	z_{out} = \gamma \hat{z} + \beta
$$



![demo3](/Users/rumimeng/Documents/paper/fig/demo3.png)



| Title                  | [Thinet: A filter level pruning method for deep neural network compression](http://openaccess.thecvf.com/content_iccv_2017/html/Luo_ThiNet_A_Filter_ICCV_2017_paper.html) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [JH Luo](https://scholar.google.com/citations?user=jiNokW0AAAAJ&hl=zh-CN&oi=sra), [J Wu](https://scholar.google.com/citations?user=0JRtCV4AAAAJ&hl=zh-CN&oi=sra), [W Lin](https://scholar.google.com/citations?user=S9g81n8AAAAJ&hl=zh-CN&oi=sra) |
| **Venue**              | CVPR 2017                                                    |
| **Model&Dataset&Code** |                                                              |
| **Type**               | `F`                                                          |

![截屏2021-01-26 下午5.33.06](/Users/rumimeng/Desktop/截屏2021-01-26 下午5.33.06.png)

| Title                  | [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1808.06866) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Yang He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+Y), [Guoliang Kang](https://arxiv.org/search/cs?searchtype=author&query=Kang%2C+G), [Xuanyi Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+X), [Yanwei Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+Y), [Yi Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Y) |
| **Venue**              | IJCAI 2018                                                   |
| **Model&Dataset&Code** |                                                              |
| **Type**               | `F`                                                          |





| Title                  | [Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1909.08174) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Zhonghui You](https://arxiv.org/search/cs?searchtype=author&query=You%2C+Z), [Kun Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+K), [Jinmian Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+J), [Meng Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+M), [Ping Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+P) |
| **Venue**              | NeurIPS 2019                                                 |
| **Model&Dataset&Code** |                                                              |
| **Type**               | `F`                                                          |

![group_gbn](fig/group_gbn.png)



| Title                  | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Jonathan Frankle](https://arxiv.org/search/cs?searchtype=author&query=Frankle%2C+J), [Michael Carbin](https://arxiv.org/search/cs?searchtype=author&query=Carbin%2C+M) |
| **Venue**              | ICLR 2019 (best)                                             |
| **Model&Dataset&Code** |                                                              |
| **Type**               | `W`                                                          |





| Title                  | [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Zhuang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Mingjie Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+M), [Tinghui Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+T), [Gao Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+G), [Trevor Darrell](https://arxiv.org/search/cs?searchtype=author&query=Darrell%2C+T) |
| **Venue**              | ICLR 2019                                                    |
| **Model&Dataset&Code** |                                                              |
| **Type**               | `F`                                                          |





| Title                  | [HRank: Filter Pruning using High-Rank Feature Map](https://arxiv.org/abs/2002.10179) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Mingbao Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+M), [Rongrong Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+R), [Yan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Yichen Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Baochang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+B), [Yonghong Tian](https://arxiv.org/search/cs?searchtype=author&query=Tian%2C+Y), [Ling Shao](https://arxiv.org/search/cs?searchtype=author&query=Shao%2C+L) |
| **Venue**              | CVPR 2020 **(Oral)**                                         |
| **Model&Dataset&Code** |                                                              |
| **Type**               | `F`                                                          |

Our HRank is inspired by the discovery that the average rank of multiple feature maps generated by a single filter is always the same, regardless of the number of image batches CNNs receive. The principle behind our pruning is that low-rank feature maps contain less information, and thus pruned results can be easily reproduced.



| Title                  | [DMCP: Differentiable Markov Channel Pruning for Neural Networks](https://arxiv.org/abs/2005.03354) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Shaopeng Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+S), [Yujie Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Quanquan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Q), [Junjie Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+J) |
| **Venue**              | CVPR 2020 **(Oral)**                                         |
| **Model&Dataset&Code** |                                                              |
| **Type**               | `F`                                                          |

![figure_pipeline](fig/figure_pipeline.png)



| Title                  | [Pruning Filter in Filter](https://arxiv.org/abs/2009.14410) |
| :--------------------- | :----------------------------------------------------------- |
| **Authors**            | [Fanxu Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Hao Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+H), [Ke Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+K), [Huixiang Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+H), [Xiaowei Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+X), [Guangming Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+G), [Xing Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+X) |
| **Venue**              | NeurIPS 2020                                                 |
| **Model&Dataset&Code** | VGG and ResNet56 on CIFAR10, ResNet18 on ImageNet            |
| **Type**               | `Stripe`                                                     |

![BrokenNet_filter](/Users/rumimeng/Documents/paper/fig/BrokenNet_filter.png)