# Markov Chain Monte Carlo

이름에 들어있는 Markov Chain과 Monte Carlo가 무엇인지 살펴보도록 하자.

1. Monte Carlo
2. Markov Chain

아주아주 간단하게 두 방법론에 대해 설명하자면 다음과 같다.

Markov chain은 우리가 관심있어 하는 어떤 분포(posterior distribution)로부터 샘플링하는 방법이다.

Monte Carlo는 이렇게 뽑은 샘플을 사용해서 기댓값을 근사하는 방법이다.

두 방법에 대해 조금 더 자세히 들엳 보도록 해보자.

## Monte Carlo Method

머신러닝에서, 분포 $p(\theta)$에 대한 기댓값 $f^*=\mathbb{E}[f(\theta)]$를 계산하는 것은 중요하다. Monte Carlo 방법은, 이러한 기댓값(integration when computing the exact value is intractable)을 근사하기 위해 사용하는 방법이다. 가장 일반적인 Monte Carlo 방법론의 아이디어는, 적분 불가능한 값을 분포 $p(\theta)$의 유한한 개수의 샘플로부터의 합으로 근사하는 아이디어이다. 구체적으로 설명하자면, $L$ samples $\{\theta^{(l)}\}^L_{l=1}$이 $p(\theta)$로 부터 샘플링 되었을 때, 우리는 기댓값을 다음과 같이 추정할 수 있다.

$$f^*\approx f_L = {1\over{L}}\sum^L_{l=1}f(\theta^{(l)})$$.

예를 들어, Bayesian inference에서는 우리는 posterior distribution으로 부터 샘플링을 해 posterior predictive distribution에 대해 근사할 수 있다.

$$p(y^\prime|x^\prime,\mathcal{D})=\mathbb{E}_{\theta|\mathcal{D}}[p(y^\prime|x^\prime,\theta)]=\int p(y^\prime|x^\prime,\theta)p(\theta|\mathcal{D})\text{d}\theta\approx{1\over{L}}\sum^L_{l=1}p(y^\prime|x^\prime,\theta^{(l)}).$$

여기서 $\theta^{(l)}\sim p(\theta|\mathcal{D})$이다.

이러한 Monte Carlo 추정은 편향되어있지 않는데, 왜냐면 $\mathbb{E}[f_L]=f^*$이기 때문이고, 이는 큰 수의 법칙으로 인해 샘플이 많아지면 많아질 수록 almost surely converge된다.

$$f_L\rightarrow f^*$$

또한 중심극한정리로 인해서 variance 는 다음과 같다.

$$\text{var}[f_L]={\text{var}[f]\over{L}}$$

따라서 샘플이 많아지면 많아질수록 높은 정확도르 얻을 수 있다.

이제 문제는 이러한 "target distribution으로부터 샘플을 어떻게 얻을 것인가 ?" 이다. 특별히, 우리가 $p(\theta)$의 unnormalized part $\tilde{p}(\theta)$에 대해 평가할 때이다. 이 때, inverse cdf method, rejection sampling, importance sampling과 같은 방법들이 있지만 해당 방법론들은 각각의 문제점이 존재한다. 따라서 Markov chain Monte Carlo가 이 문제를 해결하기 위해 등장한다.

## Markov Chains

$t=1,...,T$에서 아래의 conditional independence를 

$$
p(\theta^{(t+1)}|\theta^{(t)},...,\theta^{(1)})=p(\theta^{(t+1)}|\theta^{(t)}).
$$

만족하는 일련의 랜덤변수 $\theta^{(1)},...,\theta^{(T)}$는 first-order Markov chain이다.

우리는 위 Markov chain을 특정짓기 위해 두가지를 정의해야 한다.

1. defining the initial distribution $p(\theta^{(0)})$,
2. defining the transition probabilities $p_t(\theta^{(t+1)}|\theta^{(t)})$ for all $t$.

Markov chain은 homogeneous 혹은 time invariant라고도 불리는데, ***모든 $t$에 대해서 transition probabilities가 항상 같을 때*** 이렇게 부른다. 그리고 만약 transition probabilities가 distribution을 변화시키기 않는다면, 이러한 Markov chain의 distribution을 invariant하다고 말한다.

homogeneous Markov chain에서, a distribution $p^*(\theta)$는 아래 조건을 만족할 때, invariant하다.

$$p^*(\theta^\prime)=\sum_{\theta} {p(\theta^\prime|\theta)p^\*(\theta)}$$

MCMC에서, 우리가 원하는 것은 Markov chain이 invariant한 어떤 특정한 distribution을 가지도록 하는 것이다. 이러한 distribution은 우리가 Monte Carlo estimation을 통해 얻을 추정값을 위해 사용되는 샘플을 얻기 위한 posterior distribution이다.

이러한 ***invariant한 특정한 distribution을 얻는 한가지 확실한 방법***은, transition probability를 다음과 같은 방법을 통해서 ***detailed balance condition***을 만족시켜 주면 된다. 

$$
p(\theta^\prime|\theta)p^\*(\theta)=p(\theta|\theta^\prime)p^\*(\theta^\prime).
$$

여기서 detailed balance condition이 의미하는 것은 두 상태 $\theta$ and $\theta^\prime$ 사이에 존재하는 흐름의 비율이 항상 두 방향 모두 같게( $\theta\rightarrow\theta^\prime\:\text{and}\:\theta^\prime\rightarrow\theta$ ) 만들어 줌을 의미한다. 이러한 detailed balance조건을 만족하는 Markov chain을 ***reversible***이라고 부른다. 이것이 $p^*(\theta)$가 Markov chain의 invariant distribution으로써 가질 수 있는 충분조건임을 다음과 같이 쉽게 보일 수 있다.

$$
\sum_\theta p(\theta^\prime|\theta)p^\*(\theta)=\sum_\theta p(\theta|\theta^\prime)p^\*(\theta^\prime)=p^\*(\theta^\prime)\sum_\theta p(\theta|\theta^\prime)=p^\*(\theta^\prime).
$$

또 다른 중요한 성질은 ***ergodicity*** 이다. Markov chain은 ergodic하다고 불리는데, 만약 invariant distribution이 초기 분포(initial distribution) $p(\theta^{(0)})$이 무엇이냐에 상관 없이 $t\rightarrow\infty$일때, 수렴한다는 것을 의미한다. ergodic Markov chain은 오직 하나의 invariant distribution을 가지고, 이를 equilibrium distribution 이라고 부른다. homogeneous Markov chain은 invariant distribution과 transition probabilities에 대해 약간의 제약을 줌으로써 ergodic하다.

구체적으로, MCMC algorithm으로 생성된 샘플의 경우 모든 state(except the initial state)에서 이전 state에 의존적이기 때문에 독립이 아니다. 그런데 여기서, 높은 상관관계를 가지는 샘플은 MCMC 추정의 분산을 키우기 때문에, 좋은 MCMC sampler는 낮은 상관관계를 가지는 샘플을 생성할 수 있어야한다. 이러한 상관관계를 측정할 수 있는 방법중 하나로 auto-correlation function(ACF)라는 것을 사용한다.

$$
\rho_t={{1\over{S-t}} \Sigma^{S-t}_{s=1}(f_s-\bar{f})(f_{s+t}-\bar{f})\over{{1\over{S-1}}\Sigma^S_{s=1}(f_s-\bar{f})^2}}
$$

여기서, $\bar{f}={1\over{S}}\Sigma^S_{s=1}f_s$그리고 $t$는 time lag이다. 더 낮은 ACF값은 더 샘플러가 독립적이라는 의미이다. 또한 MCMC 방법을 통한 샘플들은 이전 state에 대해서 더 독립적일 때, 우리는 이러한 MCMC algorithm을 ***mixes better or has better mixing rate***라고 말한다.

이런 MCMC가 왜 중요하냐 ?

Energy based model과 같이 우리가 어떤 intratable한 integral을 직접 계산해서 구하는 것은 매우 비용이 많이든다. MCMC는 고차원의 데이터에서의 sampling을 가능케 하고 sampling을 통해 계산할 수 없었던 integral과 같은 것들에 대해 근사할 수 있도록 해준다. MCMC를 보면서 여러가지 sampling 기법들을 함께 알아보고 최종적으로 SGLD가 실제 고차원의 데이터에서 어떻게 사용되는지 알아보자.


- ## Rejection Sampling

In numerical analysis and computational statistics, rejection sampling is a basic technique used to generate observations from a distribution. Rejection sampling is based on the observation that to sample a random variable in one dimension, one can perform a uniformly random sampling of the two-dimensional Cartesian graph, and keep the samples in the region under the graph of its density function. Note that this property can be extended to N-dimension functions

---
- ## Importance Sampling

Importance sampling is a Monte Carlo method for **evaluating properties of a particular distribution**, while only having samples generated from a different distribution than the distribution of interest. Importance sampling is also related to umbrella sampling in computational physics. Depending on the application, the term may refer to the process of sampling from this alternative distribution, the process of inference, or both.

---
- ## Gibbs Sampling

In statistics, Gibbs sampling or a Gibbs sampler is a Markov chain Monte Carlo (MCMC) algorithm for obtaining a sequence of observations which are approximated from a specified multivariate probability distribution, when direct sampling is difficult. This sequence can be used to approximate the joint distribution (e.g., to generate a histogram of the distribution); to approximate the marginal distribution of one of the variables, or some subset of the variables (for example, the unknown parameters or latent variables); or to compute an integral (such as the expected value of one of the variables). Typically, some of the variables correspond to observations whose values are known, and hence do not need to be sampled.

Gibbs sampling is commonly used as a means of statistical inference, especially Bayesian inference. It is a randomized algorithm (i.e. an algorithm that makes use of random numbers), and is an alternative to deterministic algorithms for statistical inference such as the expectation-maximization algorithm (EM).

As with other MCMC algorithms, Gibbs sampling generates a Markov chain of samples, each of which is correlated with nearby samples. As a result, care must be taken if independent samples are desired. Generally, samples from the beginning of the chain (the burn-in period) may not accurately represent the desired distribution and are usually discarded.

---
- ## Markov Chain Monte Carlo

In statistics, Markov chain Monte Carlo (MCMC) methods comprise a class of algorithms for sampling from a probability distribution. By constructing a Markov chain that has the desired distribution as its equilibrium distribution, one can obtain a sample of the desired distribution by recording states from the chain. The more steps that are included, the more closely the distribution of the sample matches the actual desired distribution. Various algorithms exist for constructing chains, including the Metropolis–Hastings algorithm.

---
- ## Metropolis-Hastings

### Metropolis-Hastings Algorithm

MCMC의 가장 대표적인 알고리즘으로 Metropolis-Hastings라는 알고리즘이 존재한다. 이 알고리즘은 다음과 같이 작동한다. 현재 state $\theta^{(t)}$, iteration $t$ 에서, 우리는 proposal sample $\theta^\prime$을 proposal distribution $q(\theta^\prime|\theta^{(t)})$에서 부터 추출한다. 이 때, proposal distribution은 우리가 정해주면 된다. 그러고나서, proposed sample $\theta^\prime$이 accept-reject test를 통과하여 acceptance probability를 다음과 같이 계산해준다.

$$
\displaystyle{\alpha(\theta^\prime|\theta^{(t)})=\min(1,{q(\theta^{(t)}|\theta^\prime)\tilde{p}(\theta^\prime)\over{q(\theta^\prime|\theta^{(t)}\tilde{p}(\theta^{(t)}))}})}.
$$

여기서 주의해서 볼 점은, $\tilde{p}(\theta)$인데, 이 확률밀도는 unnormalized probability density이며, 우리는 이 값을 측정하는 것을 필요로 한다. acceptance probability가 주어졌을 때, 만약에 주어진 acceptance probability $\alpha$가 uniform distribution으로 부터 샘플링된 샘플보다 큰 경우, 우리는 proposed sample $\theta^\prime$을 다음 state로 채택한다. 반대의 경우에는 $\theta^{(t)}$를 다음 state로 복사한다. 수도코드는 아래의 알고리즘과 같다.

![Screenshot 2023-01-18 at 23 16 43](https://user-images.githubusercontent.com/111332590/213194471-916c2ba1-e943-4f98-8c23-c8c7f3f909e4.png)

여기서 Metropolis-Hasting algorithm은 $p(\theta)$가 invariant distribution임을 다음으로써 쉽게 보여줄 수 있다.

$$
p(\theta)q(\theta^\prime|\theta)\alpha(\theta^\prime|\theta)=\min(p(\theta)q(\theta^\prime|\theta),p(\theta^\prime)q(\theta|\theta^\prime))=\min(p(\theta^\prime)q(\theta|\theta^\prime),p(\theta)q(\theta^\prime|\theta))=p(\theta^\prime)q(\theta|\theta^\prime)\alpha(\theta|\theta^\prime)
$$

continuous space에서, proposal distribution( $q(\theta^\prime|\theta))$ )는 Gaussian distribution을 사용한다. 이 때, current state를 mean으로 설정하고 variance는 사용자가 설정한다.

이러한 특정한 알고리즘을 우리는 random walk Metropolis (RWM) algorithm이라고 부른다. RWM에서, Gaussian proposal distribution의 variance parameter를 적절하게 설정하는 것이 중요하다. 만약 작은 값을 사용하게되면 높은 acceptance rate를 가지지만, mixing rate가 poor해진다. 반대로, 큰 variance를 사용하게되면 chain이 많이 이동하기 때문에, 큰 step을 가지고 accepted된다. 그러나 acceptance rate가 작아진다.

Gaussian proposal distribution을 사용했을 때 한가지 문제점은, 우리가 실제로 target probability distribution의 gradient를 사용하여 acceptance probability를 결정하는 것과 같이 명확한 direction을 가지고 step을 이어 가는 것을 고려할 수 없다는 점이다.

---
- ## Stochastic Gradient Langevin Dynamics

Stochastic gradient Langevin dynamics (SGLD) is an optimization and sampling technique composed of characteristics from Stochastic gradient descent, a Robbins–Monro optimization algorithm, and Langevin dynamics, a mathematical extension of molecular dynamics models. Like stochastic gradient descent, SGLD is an iterative optimization algorithm which uses minibatching to create a stochastic gradient estimator, as used in SGD to optimize a differentiable objective function. Unlike traditional SGD, SGLD can be used for Bayesian learning as a sampling method. SGLD may be viewed as Langevin dynamics applied to posterior distributions, but the key difference is that the likelihood gradient terms are minibatched, like in SGD. SGLD, like Langevin dynamics, produces samples from a posterior distribution of parameters based on available data. First described by Welling and Teh in 2011, the method has applications in many contexts which require optimization, and is most notably applied in machine learning problems.



<details>
<summary>References</summary>
<div>
An Introduction to MCMC for Machine Learning

rejection sampling : https://angeloyeo.github.io/2020/09/16/rejection_sampling.html,

MCMC : https://github.com/Joseph94m/MCMC/blob/master/MCMC.ipynb

Gibbs sampling : https://jehyunlee.github.io/2021/04/20/Python-DS-69-gibbsampling/

Importance sampling : https://pasus.tistory.com/52

</div>
</details>
