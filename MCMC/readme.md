# Markov Chain Monte Carlo

이름에 들어있는 Markov Chain과 Monte Carlo가 무엇인지 살펴보도록 하자.

1. Monte Carlo
2. Markov Chain

아주아주 간단하게 두 방법론에 대해 설명하자면 다음과 같다.

Markov chain은 우리가 관심있어 하는 어떠 분포(posterior distribution)로부터 샘플링하는 방법이다.

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

이제 문제는 이러한 "target distribution으로부터 샘플을 어떻게 얻을 것인가 ?" 이다. 특별히, 우리가 $p(\theta)$의 unnormalized part $\tilde{p}(\theta)$에 대해 평가할 때이다. 이 때, inverse cdf method, rejection sampling, importance sampling과 같으 방법들이 있지만 해당 방법론들은 각각의 문제점이 존재한다. 따라서 Markov chain Monte Carlo가 이 문제를 해결하기 위해 등장한다.


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
