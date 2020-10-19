#CUDA version stripe.py

We perform stripe-wise convolution operation by python and PyTorch. Thus it can't reflect the actual inference time.

We are working on realizing a CUDA version. And you are welcome to implement this function by yourself.

We provide two ways to simply implement the function here：

$$
\begin{array}{l}
def~~Stripe\_CUDA\_Kernel(W,X,I,Y):\\
~~~~idx = threadIdx.x + blockDim.x * blockIdx.x\\
~~~~b=idx/(N^{'}\times H\times W)\\
~~~~n=idx/(H\times W)\%N^{'}\\
~~~~h=idx/W\%H\\
~~~~w=idx\%W\\
~~~~o=I[n]\%(K\times K)\\
~~~~h_{s}=h+I[n]/K\%K-\frac{K+1}{2}\\
~~~~w_{s}=w+I[n]\%K-\frac{K+1}{2}\\
~~~~Y_{o,b,h_{s},w_{s}}+=\sum_{c}^{C}W_{n,c}\times X_{c,b,h,w}\\
W\in\mathbb{R}^{N^{'}\times C},
X\in \mathbb{R}^{C\times H\times W},
I\in\mathbb{R}^{N^{'}},
Y\in \mathbb{R}^{N\times H\times W}=0\\
Stripe\_CUDA\_Kernel<<<BLOCKS,THREADS>>>(W,X,I,Y)\\
\end{array}
$$
or
$$
\begin{array}{l}
def~~Stripe\_CUDA\_Kernel(W,X,I,Y):\\
~~~~idx = threadIdx.x + blockDim.x * blockIdx.x\\
~~~~b=idx/(N^{'}\times H\times W)\\
~~~~n=idx/(H\times W)\%N^{'}\\
~~~~h=idx/W\%H\\
~~~~w=idx\%W\\
~~~~o=I[n]\\
~~~~h_{s}=h+I[n]/K\%K-\frac{K+1}{2}\\
~~~~w_{s}=w+I[n]\%K-\frac{K+1}{2}\\
~~~~Y_{o,b,h_{s},w_{s}}=\sum_{c}^{C}W_{n,c}\times X_{c,b,h,w}\\
W\in\mathbb{R}^{N^{'}\times C},
X\in \mathbb{R}^{C\times H\times W},
I\in\mathbb{R}^{N^{'}},
Y\in \mathbb{R}^{(K\cdot K\cdot N)\times H\times W}\\
Stripe\_CUDA\_Kernel<<<BLOCKS,THREADS>>>(W,X,I,Y)\\
Y=Y.reshape((K\cdot K)\times N\times H\times W).sum(0)\\
\end{array}
$$
