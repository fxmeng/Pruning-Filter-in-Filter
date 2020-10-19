# CUDA version stripe.py

We perform stripe-wise convolution operation by python and PyTorch. Thus it can't reflect the actual inference time.

We haven't finished the CUDA version code and we will working on it. you are welcome to implement this function by yourself.

We provide two ways to implement this function here for reference(open by Typora)：



$
\begin{array}{l}
def~~Stripe\_CUDA\_Kernel(W,X,I,Y):\\
~~~~idx = threadIdx.x + blockDim.x * blockIdx.x\\
~~~~b=idx/(N^{'}\times H\times W)~~~~~~~~~~~~~~~~~~~~~\%batch_idx\\
~~~~n=idx/(H\times W)\%N^{'}~~~~~~~~~~~~~~~~~~~~~\%stripe_idx\\
~~~~h=idx/W\%H\\
~~~~w=idx\%W\\
~~~~o=I[n]\%(K\times K)~~~~~~~~~~~~~~~~~~~~~\%output_channel_idx\\
~~~~h_{s}=h+I[n]/K\%K-\frac{K+1}{2}~~~~~~~~~~~~~~~~~~~~~\%h_shift\\
~~~~w_{s}=w+I[n]\%K-\frac{K+1}{2}~~~~~~~~~~~~~~~~~~~~~\%w_shift\\
~~~~Y_{o,b,h_{s},w_{s}}+=\sum_{c}^{C}W_{n,c}\times X_{c,b,h,w}\\
W\in\mathbb{R}^{N^{'}\times C},~~~~~~~~~~~~~~~~~~~~~\%Weights\\
I\in\mathbb{R}^{N^{'}},~~~~~~~~~~~~~~~~~~~~~\%Stripes'~output~channel~index~getting~from~Filter Skeleton\\
X\in \mathbb{R}^{C\times H\times W},~~~~~~~~~~~~~~~~~~~~~\%Input~feature~map\\
Y\in \mathbb{R}^{N\times H\times W}=0~~~~~~~~~~~~~~~~~~~~~\%Output~feature~map\\
Stripe\_CUDA\_Kernel<<<BLOCKS,THREADS>>>(W,X,I,Y)\\
\end{array}
$

or
$
\begin{array}{l}
def~~Stripe\_CUDA\_Kernel(W,X,I,Y):\\
~~~~idx = threadIdx.x + blockDim.x * blockIdx.x\\
~~~~b=idx/(N^{'}\times H\times W)~~~~~~~~~~~~~~~~~~~~~\%batch_idx\\
~~~~n=idx/(H\times W)\%N^{'}~~~~~~~~~~~~~~~~~~~~~\%stripe_idx\\
~~~~h=idx/W\%H~~~~~~~~~~~~~~~~~~~~~\%h_shift\\
~~~~w=idx\%W\\
~~~~o=I[n]~~~~~~~~~~~~~~~~~~~~~\%output_channel_idx\\
~~~~h_{s}=h+I[n]/K\%K-\frac{K+1}{2}~~~~~~~~~~~~~~~~~~~~~\%h_shift\\
~~~~w_{s}=w+I[n]\%K-\frac{K+1}{2}~~~~~~~~~~~~~~~~~~~~~\%w_shift\\
~~~~Y_{o,b,h_{s},w_{s}}=\sum_{c}^{C}W_{n,c}\times X_{c,b,h,w}\\
W\in\mathbb{R}^{N^{'}\times C},~~~~~~~~~~~~~~~~~~~~~\%Weights\\
I\in\mathbb{R}^{N^{'}},~~~~~~~~~~~~~~~~~~~~~\%Stripes'~output~channel~index~getting~from~Filter Skeleton\\
X\in \mathbb{R}^{C\times H\times W},~~~~~~~~~~~~~~~~~~~~~\%Input~feature~map\\
Y\in \mathbb{R}^{(K\cdot K\cdot N)\times H\times W}~~~~~~~~~~~~~~~~~~~~~\%Output~feature~map\\
Stripe\_CUDA\_Kernel<<<BLOCKS,THREADS>>>(W,X,I,Y)\\
Y=Y.reshape((K\cdot K)\times N\times H\times W).sum(0)\\
\end{array}
$

