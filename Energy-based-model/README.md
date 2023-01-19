## Energy Based Models

## Contents 
- [Your Classifier Is Secretly An Energy Based Model And You Should Treat It Like One](https://github.com/mawjdgus0812/EBM-tutorial/tree/main/Energy-based-model/JEM)

- [Energy-Based Contrastive Learning of Visual Representations](https://github.com/mawjdgus0812/EBM-tutorial/tree/main/Energy-based-model/EBCLR)

### Energy based model basic

$Y^i$ : the correct answer

$Y^{*i}$ : the answer produced by the model → the answer with the lowest energy

$\bar{Y}^i$ : the most offending incorrect answer → The answer that has the lowest energy among all the incorrect answer

1. The architecture : the internal structure of $E(W,Y,X)$
2. The inference algorithm : the method for finding a value of $Y$ that minimizes $E(W,Y,X)$ for any given $X$
3. The loss function : $\mathcal{L}(W,\mathcal{S})$ measures the quality of an energy function using the training set.
4. The learning algorithm : the method for finding a $W$ that minimizes the loss functional over the family of energy functions $\mathcal{E}$, given the training set.

---

For each of the many situations, a specific strategy, called the *inference procedure*, must be employed to find the $Y$ that minimizes $E(Y,X)$. In many real situations, the inference procedure will produce an approximate result, which may or may not be the global minimum of $E(Y,X)$ for a given $X$. In fact, there may be situations where $E(Y,X)$ has several equivalent minima. ***The best inference procedure to use often depends on the internal structure of the model***.


<details>
<summary>References</summary>
<div>
A Tutorial on Energy-Based Learning

JEM : https://github.com/wgrathwohl/JEM,

EBCLR : https://github.com/1202kbs/EBCLR

</div>
</details>
