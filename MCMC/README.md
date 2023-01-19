# Markov Chain Monte Carlo

## Contents
- [Monte Carlo](#monte-carlo-method)
- [Markov Chains](#markov-chains)
- [Rejection Sampling](#rejection-sampling)
- [Importance Sampling](#importance-sampling)
- [Markov Chain Monte Carlo](#markov-chain-monte-carlo)
- [Metropolis-Hastings Algorithm](#metropolis-hastings-algorithm)
- [Gibbs Sampling](#gibbs-sampling)
- [Stochastic Gradient Lagevin Dynamics](#stochastic-gradient-langevin-dynamics)
- References

## [Monte Carlo Method](https://github.com/mawjdgus0812/EBM-tutorial/blob/main/MCMC/Monte%20Carlo%20estimation.ipynb)

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

## [Markov Chains](https://github.com/mawjdgus0812/EBM-tutorial/blob/main/MCMC/Markov_Chains.ipynb)

$t=1,...,T$에서 아래의 conditional independence를 

$$
p(\theta^{(t+1)}|\theta^{(t)},...,\theta^{(1)})=p(\theta^{(t+1)}|\theta^{(t)}).
$$

만족하는 일련의 랜덤변수 $\theta^{(1)},...,\theta^{(T)}$는 first-order Markov chain이다.

우리는 위 Markov chain을 특정짓기 위해 두가지를 정의해야 한다.

1. defining the initial distribution $p(\theta^{(0)})$,
2. defining the transition probabilities $p_t(\theta^{(t+1)}|\theta^{(t)})$ for all $t$.

Markov chain은 임의의 initial state에서 시작해서, 다음 state로, transition probability에 의해 움직인다.

이러한 Markov chain은 memoryless라고도 불리는데, 위의 조건처럼 다음 state가 바로 이전 state에만 영향을 받는 성질을 말한다.

만약 Markov chain에서의 transition probability가 항상 같으면, homogeneous 혹은 time invariant라고도 불리는데, ***모든 $t$에 대해서 transition probabilities가 항상 같다***. 이러한 homogeneous한 성질은 chain이 $t\rightarrow\infty$로 진행되면 될수록, stationary distribution이라고 불리는 equilibrium에 도달하는 성질이다. 

$$p(\theta^{(t+1)}|\theta^{(t)})=p(\theta^{(t)}|\theta^{(t-1)})$$

그리고 만약 transition probabilities가 distribution을 변화시키기 않는다면, 이러한 Markov chain의 distribution을 invariant하다고 말한다.

homogeneous Markov chain에서, a distribution $p^*(\theta)$는 아래 조건을 만족할 때, invariant하다.

$$p^*(\theta^\prime)=\sum_{\theta} {p(\theta^\prime|\theta)p^\*(\theta)}$$

<!-- 위 식의 의미를 풀어보면, 현재 state에 대한 target distribution에 대한 trainsition probability의 기댓값이 다음 state에 대한 target distribution에 대한 기댓값과 같다는 의미이다. -->

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


- ## [Rejection Sampling](https://github.com/mawjdgus0812/EBM-tutorial/blob/main/MCMC/Rejection%20sampling.ipynb)



Rejection Sampling이란 어떤 특정 확률 분포(target density) $f(x)$에서 샘플을 추출할 때, 우리가 이러한 target function의 pdf는 알고 있지만, 그 함수에서 직접 샘플링하는 것이 어렵거나 불가능할 때 사용되는 방법이다.

- Rejection Sampling을 사용하기 위해서는 목표 확률 분포 $f(x)$의 확률밀도함수 (probability density function, PDF)를 알고 있어야 한다.
- 제안분포 $g(x)$를 설정하여 이를 이용해 샘플을 추출하고자 한다.

알고리즘은 다음과 같다.

Set $i=1$ Repeat until $i=N$
1. Sample $x^{(i)}\sim q(x)$ and $u\sim U_{(0,1)}$.
2. If $u < {f(x^{(i)})\over{Mg(x^{(i)})}}$, then accept $x^{(i)}$ and increment the counter $i$ by 1. Otherwise, reject.

![Screenshot 2023-01-19 at 4 39 21](https://user-images.githubusercontent.com/111332590/213278231-8b6b7cd9-a02f-49a6-9ad1-39d237944213.png)

---
- ## [Importance Sampling](https://github.com/mawjdgus0812/EBM-tutorial/blob/main/MCMC/Importance%20sampling.ipynb)

Rejection Samplig(기각 샘플링)의 경우, 샘플 추출시에 reject 비율이 굉장히 크다. 따라서 원하는 크기의 표본을 얻기까지 오랜시간이 걸리게 된다. 이러한 단점을 보완하는 샘플링 방법중에, wasted sample이 없도록 표본추출하는 방법을 Inportance sampling이라고 한다.

샘플링의 가장 큰 목적은 두가지이다.

1. 특정 확률밀도함수의 기댓값 계산
2. 특정 확률값 계산

따라서, 어떤 특정 값의 계산을 위해 샘플링을 한다면, 그 값을 필요로 하는 것이기 때문에 많은 표본을 추출할 필요가 없다. 즉 , 표본 추출시 버려지는 샘플이 없도록 효율적으로 샘플링을 한다는 말이다.

어떤 함수 $h(x)$에 대해서 기댓값은 다음과 같다.

$$E_f[h(x)]=\int h(x)f(x)dx$$

이 때, 원래 $f(x)$를 추출하는 것이 어려워, proposal distribution g(x)를 이용한 샘플링 방법이 존재한다. 여기서도 마찬가지로 $f(x)$에서 바로 샘플링 할 수 없기 때문에, $g(x)$에서 대신 샘플링 할 것이다.

$$E_f[h(x)]=\int h(x)f(x)dx = \int h(x){f(x)\over{g(x)}}g(x)dx=E_g[h(x){f(x)\over{g(x)}}]$$

이를 Monte Carlo Estimation을 통해 근사하면

$$E_f[h(x)]\approx {1\over{n}}\sum^n_{i=1}h(x_i){f(x_i)\over{g(x_i)}}$$

이 식을 다시 정리하면

$$E_f[h(x)]\approx {1\over{n}}\sum^n_{i=1}h(x_i){f(x_i)\over{g(x_i)}}={1\over{n}}\sum^n_{i=1}w(x_i)h(x_i) \: \text{weight:}w(x_i)={f(x_i)\over{g(x_i)}}$$

This applies when $P$ and $Q$ are both normalized

For unnormalized case

$$\mathbb{E}_{x\sim P}[f(x)]\approx{{\Sigma^n_{i=1}f(x_i){P(x_i)\over{Q(x_i)}}\over{\Sigma^n_{i=1}{P(x_i)\over{Q(x_i)}}}}}$$

---
- ## Markov Chain Monte Carlo

이름에 들어있는 Markov Chain과 Monte Carlo가 무엇인지 살펴보도록 하자.

1. [Monte Carlo](#monte-carlo-method)
2. [Markov Chains](#markov-chains)

아주아주 간단하게 두 방법론에 대해 설명하자면 다음과 같다.

Markov chain은 우리가 관심있어 하는 어떤 분포(posterior distribution)로부터 샘플링하는 방법이다.

Monte Carlo는 이렇게 뽑은 샘플을 사용해서 기댓값을 근사하는 방법이다.

예를 들어, Bayesian inference에서는 우리는 posterior distribution으로 부터 샘플링을 해 posterior predictive distribution에 대해 근사할 수 있다.

$$p(y^\prime|x^\prime,\mathcal{D})=\mathbb{E}_{\theta|\mathcal{D}}[p(y^\prime|x^\prime,\theta)]=\int p(y^\prime|x^\prime,\theta)p(\theta|\mathcal{D})\text{d}\theta\approx{1\over{L}}\sum^L_{l=1}p(y^\prime|x^\prime,\theta^{(l)}).$$

이 때, 위 식을 구하기 위해서는 posterior distribution에서의 sampling과정이 필요한데, 이러한 과정이 일반적인 sampling으로 쉽게 진행될 수 없다. 고차원의 데이터에서도 잘 작동하는 여러 MCMC sampling을 통해 이를 가능케 할 수 있다.

---
- ## [Metropolis-Hastings Algorithm](https://github.com/mawjdgus0812/EBM-tutorial/blob/main/MCMC/Metropolis-Hastings.ipynb)

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
- ## [Gibbs Sampling](https://github.com/mawjdgus0812/EBM-tutorial/blob/main/MCMC/Gibbs%20sampling.ipynb)

Gibbs sampling은 강력한 MCMC알고리즘으로, 

Gibbs sampling을 coordinate-wise sampling method로 볼 수 있다. 좀 더 구체적으로, distribution $p(\theta)=p(\theta_1,...,\theta_D)$로 부터 샘플링 하기 를 원한다고 해보자. 우리는 $\theta$의 $i$ 번째 요소를 $\theta_i$라 하고, $\theta_{\\i}$ 를 $\theta_i$를 제외한 모든 $\theta_1,..,\theta_D$라고 할 것이다. Gibbs sampling은 conditional distribution $p(\theta_i|\theta_{\\i})$를 proposal distribution으로서 사용하고, $i$를 component index로 변환시키면서 샘플을 뽑는다.

예를들어 $D=3$이라고 할때, 각 iteration $t$에서, 우리는 sampling을 다음과 같이 할 수 있다.

$$
\theta_1^{(t+1)}\sim p(\theta_1|\theta_2^{(t)},\theta_3^{(t)}),\:\theta_2^{(t+1)}\sim p(\theta_2|\theta_1^{(t+1)},\theta_3^{(t)}),\: \text{and}\:\:\theta_3^{(t+1)}\sim p(\theta_3|\theta_1^{(t+1)},\theta_2^{(t+1)})
$$

![Screenshot 2023-01-19 at 3 51 46](https://user-images.githubusercontent.com/111332590/213268855-3d58ff8b-28f8-4297-9f7e-94aa8f2ecf4a.png)

각각의 업데이트는 distribution의 invariant를 보존한다. $i$번째 component를 업데이트할 때, marginal distribution $p(\theta_{\\i})$는 바뀌지 않는다. 왜냐면 remaining variables인 $\theta_{\\i}$를 업데이트 하지 않기 때문이다. 그리고  정의에 의해 우리는 conditional distribution $p(\theta_i,\theta_{\\i})$로부터 정확히 샘플링 한다. 따라서 joint distribution은 invariant하다.

Gibbs sampling은 Metropolis-Hasting algorithm의 한 종류로 볼 수 있는데, proposal distribution이 conditional distribution인 $q(\theta_i|\theta)=p(\theta_i|\theta_{\\i})$인 경우로 볼 수 있다. 그렇다면 $\min(1, r)$에서의 ratio $r$은 항상 1이되며 따라서 언제나 proposal이 accept된다.

$$
{p(\theta^\prime_i|\theta_{\\i})p(\theta_{\\i})\over{p(\theta_i|\theta_{\\i})p(\theta_{\\i})}}\times{p(\theta_i|\theta_{\\i})\over{p(\theta^\prime_i|\theta_{\\i})}}=1.
$$

Gibbs sampling의 가장 치명적인 단점은 conditional posterior distribution으로부터 샘플링이 쉽게 되어야 한다는 점이다. (이부분 만족시키기가 어렵지 않을까)

---
- ## [Stochastic Gradient Langevin Dynamics](https://github.com/mawjdgus0812/EBM-tutorial/tree/main/MCMC/SGLD)[not yet]

Stochastic gradient Langevin dynamics (SGLD) is an optimization and sampling technique composed of characteristics from Stochastic gradient descent, a Robbins–Monro optimization algorithm, and Langevin dynamics, a mathematical extension of molecular dynamics models. Like stochastic gradient descent, SGLD is an iterative optimization algorithm which uses minibatching to create a stochastic gradient estimator, as used in SGD to optimize a differentiable objective function. Unlike traditional SGD, SGLD can be used for Bayesian learning as a sampling method. SGLD may be viewed as Langevin dynamics applied to posterior distributions, but the key difference is that the likelihood gradient terms are minibatched, like in SGD. SGLD, like Langevin dynamics, produces samples from a posterior distribution of parameters based on available data. First described by Welling and Teh in 2011, the method has applications in many contexts which require optimization, and is most notably applied in machine learning problems.



<details>
<summary> References </summary>
<div>
An Introduction to MCMC for Machine Learning

rejection sampling : https://angeloyeo.github.io/2020/09/16/rejection_sampling.html,

MCMC : https://github.com/Joseph94m/MCMC/blob/master/MCMC.ipynb

Gibbs sampling : https://jehyunlee.github.io/2021/04/20/Python-DS-69-gibbsampling/

Importance sampling : https://pasus.tistory.com/52

</div>
</details>
