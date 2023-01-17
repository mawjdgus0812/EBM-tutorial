# MCMC

위키 피디아 정의에 의하면, Markov Chain Monte Carlo, "MCMC는 마르코프 연쇄 구성에 기반한 확률 분포로부터 분포의 정적 분포(stationary)를 갖는 표본을 추출하는 알고리즘의 분류" 라고 한다. 한마디로 하면 샘플링 방법이다.

Monte Carlo

Monte Carlo는 쉽게 말하면, 평균과 분산과 같은 통계적 수치를 얻기 위해 계산이 아니라 실제로 수행하는 방법이다.

굳이 이렇게 실제로 수행하는 이유는, 통계는 무한히 많은 시도를 거쳐야지 정답에 가까워지지만, 실제로 이렇게 무한하게 시행하는건 불가능하기 때문에, 유한한 시도만으로 정답을 추정하는 작업이 필요하다. 그리고 그러한 방법 중 하나가 Monte Carlo방법이다.

가장 흔한 예시로, 원의 넓이를 구하는 예시가 있는데,

정사각형 안에 무수히 많은 점을 찍으면서, 중심으로부터 거리가 1이하면 빨간색, 아니면 파란색으로 칠해 줌으로써, 전체적으로 찍은 점의 개수와 빨간색으로 찍힌 점의 개수의 비율을 계산해서 원래 사각형 면적을 곱해주면 원의 넓이를 대략적으로 추정할 수 있다. 

Markov Chain

Markov Chain은 어떤 상태에서 다른 상태로 넘어갈 때, 바로 전 단계의 상태에만 영향을 받는 확률 과정을 의미한다.

어제 비가왔을 때 오늘 비가 올 확률이 이만큼.
어제 비가 안오면 오늘 비가 올 확률이 이만큼.

처럼 어제 상태에만 영향을 받는 과정을 마르코프 성질(Markov property)를 가진다고 하며, 이러한 확률 과정을 Markov chain이라 한다.
MCMC는 샘플링 방법줌, 가장 마지막에 뽑힌 샘플이 다음번 샘플에 영향을 주는 의미에서 Markov chain이 들어갔다고 볼 수 있다.

이런 MCMC가 왜 중요하냐 ?

Energy based model과 같이 우리가 어떤 intratable한 integral을 직접 계산해서 구하는 것은 매우 비용이 많이든다. MCMC는 고차원의 데이터에서의 sampling을 가능케 하고 sampling을 통해 계산할 수 없었던 integral과 같은 것들에 대해 근사할 수 있도록 해준다.


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
