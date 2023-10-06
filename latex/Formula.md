## Bayes' theorem
$$
P(v|d_{obs})\propto P(d_{obs}|v)\cdot P(v)
$$

## Likelihood
$$
P(d_{obs}|v) \propto e^{\frac{-\sum_{i=1}^{n} (d_{obs}-P(v))}{2\sigma ^{2}}}
=e^{-\frac{J(v)}{\sigma ^{2}}}\\
$$

## Prior
$$
P(v)\propto e^{-\frac{1}{2}v^{t}Qv}
$$

## Posterior
$$
-log^{P(v|d_{obs})}\propto \frac{J(v)}{\sigma ^{2}}+\frac{1}{2}v^{t}Qv\\

\widehat{v}_{MAP}(d_{obs}) = \underset{v}{argmin}\left\{ \frac{J(v)}{\sigma ^{2}}+\frac{1}{2}v^{t}Qv\right\}
$$

