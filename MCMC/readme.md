MCMC

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
