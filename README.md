---
title: Quantifying Uncertainties in Variational Bayesian Machine Learning - Application to 2D Seismic Imaging
date: today
author: me
---

## **Quantifying Uncertainties in Variational Bayesian Machine Learning: Application to 2D Seismic Imaging**

Objective of the project: to obtain the posterior distribution of the velocity model by implementing a variational bayesian using the gradient decent

***Keywords:***
>Full waveform inversion, Inverse problem, Variational Autoencoder (vae), Deep Learning


## Reparametrization

The reparametrizaion transform function $f$ is defined as:

$$ v = f(u) = a + \frac{b-a}{1+\exp(-u)} $$

>where $u$ is the unconstrained parameter space and $v$ is the physical parameter space. The constants $a$ and $b$ represent the minimum and maximum values for the physical parameter, respectively. The function maps the unconstrained space $u$ to the physical parameter space $v$ using the sigmoid function $\frac{1}{1+\exp(-u)}$ that squashes the values of $u$ to the range between 0 and 1. The squashed values are then scaled by $(b-a)$ and shifted by $a$ to obtain the values in the physical parameter space $v$.

The inverse transformation function is given by:

$$ u = f^{-1}(v) = log(v - a) - log(b - v) $$

The inverse of the transformation function is used to obtain the unconstrained parameter u from the physical parameter v. 


## Objective function

The goal is to optimize the variational parameters $\mu$ and $\omega$ to minimize the Kullback-Leibler (KL) divergence between the true posterior and the approximating distribution. The ELBO is given by:

### KL divergence is defined as:

$$KL(q(v) || p(v|d_{obs})) = E_q[\log \frac{q(v)}{p(v|d_{obs})}]$$

* Using Bayes' theorem, we can express the true posterior as:

$$p(v|d_{obs}) = \frac{p(d_{obs}|v)p(v)}{p(d_{obs})}$$

* Substituting this into the definition of the KL divergence gives:

$$KL(q(v) || p(v|d_{obs})) = E_q[\log \frac{q(v)}{p(d_{obs}|v)p(v)/p(d_{obs})}]$$

$$= E_q[\log \frac{q(v)}{p(d_{obs}|v)p(v)}] + \log p(d_{obs})$$

### The objective function is defined as:

$$\text{ELBO} = \mathbb{E}{q(v)}[\log p(d{obs}|v)] + \mathbb{E}{q(v)}[\log p(v)] - \mathbb{E}{q(v)}[\log q(v)] - \log p(d_{obs}) $$

$$= \mathbb{E}{q(v)}[\log p(d{obs}|v)] - \mathbb{E}{q(v)}[\log \frac{q(v)}{p(v)}]- C $$

$$ = \mathbb{E}{q(v)}[\log p(d{obs}|v)] - {KL}(q(v) || p(v))$$

>where $\log p(d_{obs}|v)$ is the log-likelihood of the data given the latent variables, $\log p(v)$ is the prior distribution over the latent variables, $\log q(v)$ is the log of the variational distribution, and $\log p(d_{obs})$ is a constant term that does not depend on $v$ and can be ignored during optimization.



## Variational Parameters
The Gaussian variational family is defined by:

$$ q(u) = \mathcal{N}(u | \mu, \Sigma) $$ 
>where $u$ represents the model parameters, and $\mu$ and $\Sigma$ are the variational mean and covariance that approximate the true posterior distribution of $u$.

To make the computation of the posterior tractable, we can use a factorized Gaussian approximation, which assumes that the different elements of $u$ are independent and have variances determined by the values in $\omega$. 

$$ q(u) = \mathcal{N}(u | \mu, \text{diag}(\exp(\omega)^2)) $$
>where the function $\exp(\omega)$ transforms the unconstrained values of $\omega$ to positive values, which represent the standard deviations of the elements of $u$. Finally, taking the square of these standard deviations results in the diagonal matrix $\text{diag}(\exp(\omega)^2)$, which represents the covariance matrix of $q(u)$.

"In factorized Gaussian variational approximation (also known as mean field approximation), the joint distribution of parameters is approximated as a product of independent univariate Gaussian distributions, which effectively neglects any correlation between parameters. This simplification allows for easier computation and has been found to be effective in practice for high-dimensional problems, although it may not capture the full complexity of the true posterior distribution."

## Gradients
The gradient of the ELBO with respect to the variational parameters $\mu$ and $\omega$ can be computed using the following formulas:

$$ \nabla_{\mu} \text{ELBO} = \frac{\partial}{\partial \mu} \mathbb{E}q [\log p(d{obs} | u)] - \nabla_{\mu} \text{KL}(q(u) || p(u)) $$

The gradient of the ELBO with respect to the unconstrained parameters $\omega$ can be computed using the chain rule:

$$ \nabla_{\omega} \text{ELBO} = \frac{\partial \text{ELBO}}{\partial \text{diag}(\Sigma)} =  \frac{\partial \text{ELBO}}{\partial \text{diag}(\exp(\omega)^2)}\frac{\partial \text{diag}(\exp(\omega)^2)}{\partial \omega}$$

$$ = \frac{\partial}{\partial \omega} \mathbb{E}q [\log p(d{obs} | u)] - \nabla_{\omega} \text{KL}(q(u) || p(u))$$

>where $\nabla_{\mu}$ and $\nabla_{\omega}$ represent the gradients with respect to $\mu$ and $\omega$, respectively. The first term in each formula represents the gradient of the expected log-likelihood, which can be estimated using Monte Carlo integration. The second term represents the gradient of the KL divergence, which has a closed-form expression.
