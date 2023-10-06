To explain the MCMC approach for obtaining the posterior distribution, let's start with Bayes' theorem:

$$ P(\mathbf{m}|\mathbf{d}) = \frac{P(\mathbf{d}|\mathbf{m})P(\mathbf{m})}{P(\mathbf{d})} $$

where $\mathbf{m}$ is the velocity model, $\mathbf{d}$ is the observed data, $P(\mathbf{m}|\mathbf{d})$ is the posterior distribution, $P(\mathbf{d}|\mathbf{m})$ is the likelihood function, $P(\mathbf{m})$ is the prior distribution, and $P(\mathbf{d})$ is the evidence.

In seismic inversion, we typically assume that the likelihood function is a Gaussian distribution:

$$ P(\mathbf{d}|\mathbf{m}) = \frac{1}{\sqrt{(2\pi)^N|\mathbf{C}|}}exp\left(-\frac{1}{2}(\mathbf{d}-\mathbf{f}(\mathbf{m}))^T\mathbf{C}^{-1}(\mathbf{d}-\mathbf{f}(\mathbf{m}))\right) $$

where $\mathbf{f}(\mathbf{m})$ is the forward model that predicts the data, $\mathbf{C}$ is the data covariance matrix, and $N$ is the number of data points.

The prior distribution $P(\mathbf{m})$ represents our prior knowledge or assumptions about the velocity model before we see the data. This is typically represented as a multivariate normal distribution:

$$ P(\mathbf{m}) = \frac{1}{\sqrt{(2\pi)^k|\mathbf{\Sigma}|}}exp\left(-\frac{1}{2}(\mathbf{m}-\mathbf{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{m}-\mathbf{\mu})\right) $$

where $\mathbf{\mu}$ is the mean vector and $\mathbf{\Sigma}$ is the covariance matrix.

To obtain the posterior distribution, we need to sample from the joint distribution $P(\mathbf{m},\mathbf{d})$. However, this is often intractable due to the high dimensionality of the problem and the complexity of the forward model. Therefore, we use Markov Chain Monte Carlo (MCMC) methods to generate samples from the joint distribution.

The idea behind MCMC is to construct a Markov chain that has the joint distribution $P(\mathbf{m},\mathbf{d})$ as its stationary distribution. This means that if we run the Markov chain for long enough, the samples will be distributed according to the joint distribution.


In the context of Bayesian FWI, the acceptance probability for the proposed model (or velocity model) $\mathbf{m}^*$ based on the current model $\mathbf{m}$ can be expressed as:

$$\alpha(\mathbf{m}^,\mathbf{m}) = \min\left(1, \frac{\pi(\mathbf{m}^|\mathbf{d})p(\mathbf{m}^)q(\mathbf{m}|\mathbf{m}^)}{\pi(\mathbf{m}|\mathbf{d})p(\mathbf{m})q(\mathbf{m}^*|\mathbf{m})}\right)$$

where:

$\pi(\mathbf{m}|\mathbf{d})$ is the posterior distribution of the model given the observed data $\mathbf{d}$.
$p(\mathbf{m})$ is the prior distribution of the model.
$q(\mathbf{m}|\mathbf{m}^*)$ is the proposal distribution for the current model given the proposed model.
$\pi(\mathbf{m}^*|\mathbf{d})$ is the posterior distribution of the proposed model given the observed data.
$p(\mathbf{m}^*)$ is the prior distribution of the proposed model.
$q(\mathbf{m}^*|\mathbf{m})$ is the proposal distribution for the proposed model given the current model.
The Metropolis-adjusted Langevin algorithm (MALA) is a variant of the Metropolis-Hastings algorithm that uses a Langevin diffusion process to generate proposals for the new state. The proposal distribution $q(\mathbf{m}^*|\mathbf{m})$ in MALA is guided by the gradient of the log-posterior distribution, resulting in more efficient and effective sampling compared to other proposal distributions such as random walk or Gaussian.

 