### YOUR CLASSIFIER IS SECRETLY AN ENERGY BASED MODEL AND YOU SHOULD TREAT IT LIKE ONE

http://arxiv.org/abs/1912.03263

Energy-based-models은 $\mathbf{x\in}\mathbb{R}^D$에 대한 probability density $p(\mathbf{x})$에 의존하며 다음과 같이 나타낼 수 있다.

$$
p_\theta(\mathbf{x})={\exp(-E_\theta(\mathbf{x}))\over{Z(\theta)}}
$$

이 때, $E_\theta(\mathbf{x}):\mathbb{R}^D\rightarrow\mathbb{R}$,는 energy function으로 알려져 있으며, 각 데이터 포인트를 scalar로 변환해준다. 그리고 $Z(\theta)=\int_\mathbf{x}\exp(-E_\theta(\mathbf{x}))$는 partition function으로 알려진 normalizing constant (with repect to $\mathbf{x}$)이다. 따라서, EBM은 $\mathbf{x}$라는 데이터 인풋을 취급하는 어떤 function 을 사용해서 scalr로 바꿔주는 과정이다.

보통 어떤 $E_\theta$를 가져가더라도, $Z(\theta)$를 계산하는 일은 매우 어려운 일이다. 따라서 우리가 흔히 사용하는 일반적인 maximum likelihood estimation은 적용하기 어렵다. 따라서 EBM을 훈련시키는 방법을 통해 해결할 수 있다. 하나의 example $\mathbf{x}$에 대해서 parameter $\theta$에 대한 log-likelihood의 미분을 다음과 같이 나타낼 수 있다.

$$
{\partial\log p_\theta(\mathbf{x})\over{\partial\theta}}=\mathbb{E}_{p_\theta(\mathbf{x}^\prime)}[{\partial E_\theta(\mathbf{x}^\prime)\over{\partial\theta}}]-{\partial E_\theta(\mathbf{x})\over{\partial\theta}}
$$

model distribution에 대한 기댓값으로 나타낼 수 있다.

안타깝게도, 우리는 $p_\theta(\mathbf{x})$에 대한 샘플은 쉽게 추출할 수 없다. 그래서 우리는 MCMC방법을 통해 이 gradient estimator를 사용할 수 밖에 없다. 이 방법은 초창기 EBM에서 많이 쓰였으며, Restricted Boltzmann Machines은 block Gibbs sampler를 사용해서 훈련을 가능케 했다.

긴 시간이 지났지만, 많은 발전이 있진 않았는데, 최근에 이러한 방법을 통해서 large-scale EBMs을 deep neural networks에 의해 parameterized된 고차원의 데이터에서 훈련하는 연구가 진행됐다. 여기서 위 식을 근사하기 위해 Stochastic Gradient Langevin Dynamics(SGLD)를 이용해 샘플링을 진행하였다.

$$
\mathbf{x}_0\sim p_0(\mathbf{x}),
$$

$$
\mathbf{x}_{i+1}=\mathbf{x}_i-{\alpha\over{2}}{\partial E_\theta(\mathbf{x}_i)\over{\partial\mathbf{x}_i}}+\epsilon,
$$

$$
\epsilon\sim\mathcal{N}(0,\alpha)
$$

여기서 $p_0(\mathbf{x})$는 일반적으로 input domain에 대한 Uniform distribution이며, step-size $\alpha$는 polynomial schdule을 통해 decayed된다. 실제로는 step-size는 standard deviation of $\epsilon$이고 빠른 훈련을 위해 biased sampler를 부분적으로 이끌 수 있도록 선택된다.

---

논문의 key observation은 $f_\theta$로 부터 얻어지는 logits에 대해 $p(\mathbf{x},y)$와 $p(\mathbf{x})$를 정의하기 위해 아주 약간 재 정의한다는 점이다. $f_\theta$에 대한 변화 없이, 데이터 포인트 $\mathbf{x}$와 label $y$간의 joint probability의 energy based model을 정의하기 위해 logits을 다시 사용한다.

$$
p_\theta(\mathbf{x},y)={\exp(f_\theta(\mathbf{x})[y])\over{Z(\theta)}}
$$

여기서 $Z(\theta)$는 알 수 없는 normalizing constant이며, $E_\theta(\mathbf{x},y)=-f_\theta(\mathbf{x})[y]$이다.

$y$를 marginalizing 하기 위해서, 다음과 같은 unnormalized density model을 구할 수 있다.

$$
p_\theta(\mathbf{x})=\sum_y p_\theta(\mathbf{x},y)={\sum_y\exp(f_\theta(\mathbf{x})[y])\over{Z(\theta)}}
$$

위 식은 이제 아무 classifier의 logits의 $\text{LogSumExp}(\cdot)$으로 energy function을 정의하기 위해서 datapoint $\mathbf{x}$에 대하여 재사용할 수 있다.

$$
E_\theta(\mathbf{x})=-\text{LogSumExp}_y(f_\theta(\mathbf{x})[y])=-\log\sum_y\exp(f_\theta(\mathbf{x})[y])
$$

임의의 scalar로부터온 logits $f_\theta(\mathbf{x})$의 이동(shifting)이 모델에 영향을 주지 못하는 일반적인 classifier와는 다르게, 위 framework에서 데이터 포인트에 대한 logits의 이동(shifting)은 데이터 포인트 $\mathbf{x}$에 대해 $\log p_\theta(\mathbf{x})$의 영향을 준다. 따라서, 이 프레임워크에서는 입력 데이터에 대한 density function으로써 사용할 뿐만 아니라 데이터와 label에 대한 joint density 에 대해서 정의하기 위해 logits을 사용함으로써 발생할 수 있는 숨겨진 자유도를 사용할 수 있도록 만들어낸다. 최종적으로 $p_\theta(\mathbf{x},y)/p_\theta(\mathbf{x})$를 계산함으로써  $p_\theta(y|\mathbf{x})$를 구함으로써, normalizing constant가 없어지고, standard Softmax parameterization을 창출한다. 따라서 이 모델은 모든 discriminative model에 있는 숨겨진 generative model을 발견했다고 할 수 있다.

---


