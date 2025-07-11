# Introduction

This is a summary of the chapter "Probabilistic Learning" from the book ["Physics Based Deep learning"](https://physicsbaseddeeplearning.org/).
## Bayes' Theorem

$$p(x|y) = \frac{p(y|x)p(x)}{p(y)}$$

that is equivalent to

$$p(x|y) p(y) = p(y|x) p(x)$$

where

$$p(x|y) p(y) = p(x, y)$$

## Non deterministic models

Oftentimes we treat a target function $f$ as deterministic. That is for an input $x$ we have a single output $y = f(x)$. However, in many cases the output is not deterministic, and we have a multidude of solution that are sampled from a distribution $\mathbf{Y}$, where the probability of a solution $y$ is given as $p(y)$. 

## Inverse Problem

With an inverse problem, the task is to find the input $x$ given the output $y$. Instead of finding a single solution, it is oftentimes more useful to find a distribution of solutions for a given output. 

For the target function $f$ we can define a distribution of solutions as $p(y|x)$, which is the probability of a solution $y$ given an input $x$. What we want to know is the distribution of the posterior $p(x|y)$

$$
p(x|y) = \frac{p(y|x)p(x)}{\int{p(y|x^\prime)p(x^\prime)dx^\prime}}
$$

where the denominator is the evidence $p(y)$, which is the probability of the output $y$. Using deep learning, we learn a conditional density estimator $q_0(y|x)$ that allows us to sample from the distribution of solutions $p(y|x)$.

