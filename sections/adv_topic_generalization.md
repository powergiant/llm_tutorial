# Generalization of Deep Learning

## Outline

As discussed in the basic machine learning section, learning starts from an unknown data-generating distribution $\mathcal{D}$ over examples $(x,y)$. A model $f_\theta$ is trained by minimizing the empirical risk
$$
\hat R_n(\theta)=\frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x_i),y_i),
$$
but the real object of interest is the population risk
$$
R(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f_\theta(x),y)].
$$
Generalization asks why a parameter $\theta$ found by training can have small $R(\theta)$, even when the network has enough parameters to interpolate the training set and even to fit random labels.

1. What generalization asks

    We begin by separating empirical risk from population risk and by stating the generalization gap, and we also introduce the classical generalization bound. This section also distinguishes expressiveness, optimization, and generalization, and explains why the core question is not merely whether a network can interpolate the data, but why the trained interpolating solution can still achieve low population risk.

2. What changed in deep learning

    Classical generalization theory controls a hypothesis class through VC dimension, Rademacher complexity, covering numbers, margins, norms, or stability. Deep learning changes the picture because modern networks are heavily overparameterized, can fit random labels, and still often generalize well, so the relevant notion of complexity must depend on the learned solution, the training algorithm, and the regime rather than on parameter count alone.

3. Experimental phenomenology

    The starting point is empirical. Networks can fit random labels, small-batch and large-batch training can reach different interpolating solutions, and optimizer, schedule, normalization, augmentation, and training time all affect which solution is selected. These observations motivate flatness, implicit regularization, and landscape-based explanations, while also showing why a single scalar complexity measure is unlikely to explain every regime.

4. Implicit regularization of optimization

    Overparameterized training has many empirical minimizers, so the algorithm matters. Gradient descent, SGD noise, Adam, weight decay, normalization, sharpness-aware minimization, stochastic weight averaging, and learning-rate schedules all bias the selected solution. We also introduce classical implicit-regularization behavior in statistical models, where this bias can often be characterized exactly through minimum-norm or max-margin behavior; in deep networks the picture is less complete but still central.

5. Flatness

    * Curvature, flatness, and measurement

        A flat minimum is a solution whose loss changes slowly under parameter perturbations. In practice, people measure top Hessian eigenvalues, Hessian trace, spectral density, perturbation sharpness, local entropy, normalized sharpness, or filter-normalized visualizations. These quantities are related through the quadratic approximation, but they are not identical, and raw parameter-space sharpness is not invariant under common neural-network reparameterizations.

    * Stationary phase, Laplace evidence, and the Hessian

        The cleanest mathematical route from curvature to model complexity is the asymptotic expansion of an integral over parameters. Around a regular nondegenerate minimum, stationary phase or Laplace's method gives an Occam factor involving $\det H$, where $H$ is the Hessian of the empirical loss. This explains why wide basins can have higher Bayesian evidence, while also showing why singular and degenerate deep-network minima require extra care.

    * PAC-Bayes, perturbation stability, and scale-sensitive bounds

        A second theoretical route is to bound the risk of a distribution over parameters. If perturbing a trained model does not increase empirical loss much, and if the perturbation distribution remains close to a prior, PAC-Bayes gives a path from flatness to generalization. Related margin-, norm-, and perturbation-based bounds replace raw parameter count by scale-sensitive quantities that better reflect the learned solution.

6. Implicit low-dimensional effects

    Deep learning models often behave as though training occurs in a much smaller effective space than their full parameter dimension suggests. Evidence for this appears first in the geometry of the loss landscape: Hessian spectra typically show many near-zero directions alongside a small number of dominant outliers, indicating that only a limited set of directions strongly influence optimization. Consistent with this, training can sometimes succeed even when restricted to random low-dimensional subspaces, and fine-tuning large models frequently requires only low-rank or otherwise low-dimensional updates. Together, these observations link ideas of flatness, representation learning, and parameter-efficient adaptation. This perspective is reinforced by results in pruning, sparsity, and compression, which suggest that a trained network often contains a smaller subnetwork capable of preserving performance. From a theoretical standpoint, compression-based generalization bounds further support this view by showing that predictors describable with fewer bits tend to generalize better. Empirically, findings such as the lottery ticket hypothesis and Hessian-aware pruning make the relationship between redundancy, curvature, and effective dimension increasingly explicit.

7. More on the shape of the loss landscape

    Local curvature and dimension of local minima are only one part of landscape geometry. The global landscape includes mode connectivity, low-loss paths between independently trained models, low-loss manifolds, permutation symmetries, barriers, and the geometry induced by normalization and residual connections. This section studies the shape of the loss landscape as a global object rather than reducing it to one Hessian at one solution.

8. Robustness, shift, and caveats

    Generalization measures must be interpreted with care. Raw flatness can be changed by reparameterization without changing the represented function; many measures correlate with test error only under restricted experimental protocols; and robustness, calibration, and distribution-shift performance are not automatic consequences of clean test generalization. A good survey therefore states the training regime carefully and distinguishes correlation, bound, and causal mechanism.

## 1. What Generalization Asks

Learning is based on a sample
$$
S=\{(x_i,y_i)\}_{i=1}^n \sim \mathcal D^n,
$$
and training produces parameters $\theta_S$ by minimizing or nearly minimizing the empirical risk $\hat R_n$. The core question is whether the same parameters also have small population risk $R(\theta_S)$. The basic object is therefore the generalization gap
$$
R(\theta_S)-\hat R_n(\theta_S),
$$
or more generally the difference between population and empirical performance of the learned predictor.

Classical learning theory often studies this gap uniformly over a class $\mathcal F$:
$$
\sup_{f\in\mathcal F}|R(f)-\hat R_n(f)|.
$$
This viewpoint leads to VC-dimension, Rademacher-complexity, covering-number, margin, norm, and stability bounds. These are powerful when the effective complexity of $\mathcal F$ is small relative to $n$. The difficulty in deep learning is that the raw architecture class is usually enormous, often large enough to interpolate the data exactly.

The challenge was made concrete by [Zhang, Bengio, Hardt, Recht, and Vinyals 2017](https://research.google/pubs/understanding-deep-learning-requires-rethinking-generalization/), who showed that standard deep networks can fit random labels while still generalizing well on real labels. This means that the question is not merely whether the model class is expressive enough to interpolate. There are three distinct issues: expressiveness asks whether the class can represent the target, optimization asks whether training can find low empirical loss, and generalization asks why the particular trained solution has low population risk.

A good survey therefore treats generalization as a regime-dependent question. Some theories explain kernel-like training near initialization, some explain margin growth in separable classification, some explain compressed or sparse predictors, and some explain Bayesian local volume around a minimum. The point is not to force one universal mechanism, but to say clearly which mechanism is active under which assumptions.

## 2. What Changed in Deep Learning

Classical theory usually controls complexity at the level of the hypothesis class. Deep learning changed the picture because modern models are heavily overparameterized, often have far more parameters than training examples, and can interpolate the data with zero training error. In that regime, parameter count alone is too crude to explain test performance.

The modern view is that effective complexity depends on the learned solution and on the training procedure that selected it. Two networks with the same architecture and the same training error can have very different test error if they are trained with different optimizers, schedules, regularizers, normalization layers, augmentation pipelines, or pretraining. This is why norms, margins, stability, compression length, effective rank, sharpness, and posterior volume all appear in the literature: they are attempts to replace raw size by solution-dependent complexity.

Deep learning also departs from classical fixed-feature settings because the representation itself changes during training. In a kernel method, the feature map is fixed and only the readout is learned. In a deep network, the internal representation, the geometry of the loss, and the final classifier are all co-adapted. The relevant complexity may therefore depend on feature learning, not only on the final linear separator in the top layer.

This shift also explains why interpolation no longer means what it did in older statistical intuition. In overparameterized regimes, interpolation can coexist with good test error, and test error can even improve past the interpolation threshold, as emphasized in the double-descent literature such as [Belkin, Hsu, Ma, and Mandal 2019](https://www.pnas.org/doi/10.1073/pnas.1903070116). The right lesson is not that classical theory was useless, but that the relevant complexity measure in deep learning is more local, more algorithm-dependent, and more regime-specific than parameter counting suggests.

## 3. Experimental Phenomenology

The starting point for modern generalization theory is empirical. Deep networks can fit random labels, yet on natural data they often generalize well. Different training protocols can also reach different interpolating solutions from the same architecture. Batch size, optimizer, learning-rate schedule, normalization, augmentation, training time, and pretraining all affect which solution is selected.

Flatness entered the story partly through this phenomenology. [Hochreiter and Schmidhuber 1997](https://direct.mit.edu/neco/article/9/1/1/6027/Flat-Minima) argued that flatter minima should generalize better because they are less sensitive to parameter perturbations. [Keskar et al. 2017](https://openreview.net/forum?id=H1oyRlYgg) reported that some large-batch runs converge to sharper solutions with worse test error than small-batch runs. Later work such as [Hoffer, Hubara, and Soudry 2017](https://papers.nips.cc/paper/6770-train-longer-generalize-better-closing-the-generalization-gap-in-large-batch-training-of-neural-networks) showed that this picture is not universal: large-batch training can generalize well once learning-rate schedules, warmup, update count, and regularization are adjusted.

Several practical methods make the selection effect explicit. [Chaudhari et al. 2017](https://openreview.net/forum?id=B1YfAfcgl) proposed Entropy-SGD, which favors regions of low loss with large local volume. [Izmailov et al. 2018](https://proceedings.mlr.press/v80/izmailov18a.html) introduced stochastic weight averaging, which often lands in wider basins with improved test error. [Foret et al. 2021](https://openreview.net/forum?id=6Tm1mposlrM) proposed sharpness-aware minimization (SAM), which explicitly penalizes local worst-case loss increase.

At the same time, the empirical literature warns against one-number explanations. [Jiang et al. 2020](https://openreview.net/forum?id=SJgIPJBFvH) compared many proposed measures and found that their predictive power depends strongly on protocol and architecture. The durable conclusion is that experimental phenomenology points to multiple interacting mechanisms. Flatness matters in some settings, margins matter in others, data augmentation can dominate both, and pretraining changes the whole regime. A realistic theory should begin from this heterogeneity rather than try to erase it.

## 4. Implicit Regularization of Optimization

In an overparameterized model there are typically many parameters with nearly identical empirical risk, including many interpolating solutions with $\hat R_n(\theta)=0$. Training therefore does more than minimize the objective. It selects one solution from a large set, and that selection bias can itself act as a regularizer.

In simple models this bias can often be characterized exactly. Gradient descent on least squares from small initialization converges to the minimum-Euclidean-norm interpolant. On separable logistic regression, gradient descent does not converge to a finite minimizer, but its direction converges to the max-margin classifier, as shown by [Soudry, Hoffer, Nacson, Gunasekar, and Srebro 2018](https://jmlr.org/papers/v19/18-188.html). Related work such as [Gunasekar et al. 2018](https://proceedings.mlr.press/v80/gunasekar18a.html) shows analogous implicit biases toward low-rank or norm-controlled solutions in matrix factorization and homogeneous models.

For deep networks, the picture is less complete but the same principle applies. Initialization scale, weight decay, normalization, data augmentation, label smoothing, early stopping, optimizer choice, and schedule all change which interpolating solution is reached. SGD noise depends on batch size, learning rate, and gradient covariance, so changing batch size can change the selected basin even when the architecture and dataset are fixed. Bayesian interpretations such as [Mandt, Hoffman, and Blei 2017](https://jmlr.org/papers/v18/17-214.html) and [Smith and Le 2018](https://openreview.net/forum?id=BJij4yg0Z) are not exact descriptions of real training, but they capture an important idea: stochastic optimization is biased toward some regions of parameter space more than others.

Some methods make that bias explicit. SAM approximately solves
$$
\min_\theta \max_{\|\epsilon\|\le \rho}\hat R_n(\theta+\epsilon),
$$
so the update prefers solutions that remain good under local perturbations. SWA averages weights along the SGD trajectory and often moves the final point toward the center of a low-loss valley. The main conceptual lesson is that in deep learning generalization is partly a property of the optimizer and schedule, not only of the architecture or the training objective.

## 5. Flatness

### Curvature, flatness, and measurement

Let $\theta_\star$ be a trained solution and let
$$
H=\nabla^2 \hat R_n(\theta_\star)
$$
be the Hessian of the empirical risk. Near a stationary point, a second-order expansion gives
$$
\hat R_n(\theta_\star+\delta)-\hat R_n(\theta_\star)\approx \frac{1}{2}\delta^\top H\delta.
$$
This formula explains several common flatness measures. The worst-case loss increase in a small Euclidean ball is governed by $\lambda_{\max}(H)$, the average increase under isotropic Gaussian perturbations is governed by $\operatorname{tr}(H)$, and local-volume calculations involve $\log\det H$ or a pseudo-determinant after removing nearly flat directions. Empirical Hessian studies such as [Sagun et al. 2017](https://arxiv.org/abs/1706.04454) and [Ghorbani, Krishnan, and Xiao 2019](https://proceedings.mlr.press/v97/ghorbani19b.html) typically find a few large outliers and a broad near-zero bulk, which suggests that the geometry is highly anisotropic rather than captured by one scalar.

The main warning is that raw parameter-space sharpness is not invariant. [Dinh, Pascanu, Bengio, and Bengio 2017](https://proceedings.mlr.press/v70/dinh17b.html) showed that ReLU rescalings can make a minimum look arbitrarily sharp without changing the represented function. This is why later work studies normalized sharpness, filter-normalized visualizations, adaptive perturbations, Fisher geometry, function-space perturbations, or margin-normalized measures instead of raw Hessian values alone.

### Stationary phase, Laplace evidence, and the Hessian

The cleanest mathematical route from flatness to complexity comes from Bayesian evidence. Consider
$$
Z_n(\beta)=\int_\Theta \exp(-\beta n\hat R_n(\theta))\pi(\theta)\,d\theta,
$$
with prior $\pi$ and inverse temperature $\beta$. Around a regular nondegenerate minimum $\theta_j$, Laplace's method gives
$$
Z_n(\beta)\approx \exp(-\beta n\hat R_n(\theta_j))\pi(\theta_j)\left(\frac{2\pi}{\beta n}\right)^{p/2}\frac{1}{\sqrt{\det H_j}}.
$$
The factor $(\det H_j)^{-1/2}$ is the Occam factor: among minima with similar empirical loss, a wider basin contributes more posterior mass. This is the formal content of the claim that flatter minima can be preferred by Bayesian model comparison, as in [MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.448) and [Hinton and van Camp 1993](https://dl.acm.org/doi/10.1145/168304.168306).

Deep networks, however, are often singular rather than regular. They have permutation symmetries, scale symmetries, redundant parameters, and degenerate Hessians. In that setting the naive Laplace approximation can fail, and singular-learning theory such as [Watanabe 2013](https://www.jmlr.org/papers/v14/watanabe13a.html) becomes relevant. So Hessian-based evidence is informative, but it is a local regular approximation unless singular geometry is treated explicitly.

### PAC-Bayes, perturbation stability, and scale-sensitive bounds

PAC-Bayes gives another route from flatness to generalization by studying a posterior distribution $Q$ over parameters rather than a single point estimate. A typical PAC-Bayes bound controls population risk by empirical risk plus a complexity term involving $\mathrm{KL}(Q\|P)$ relative to a prior $P$. If $Q$ is centered at $\theta_\star$ with covariance $\Sigma$, then
$$
\mathbb{E}_{\delta\sim\mathcal N(0,\Sigma)}\hat R_n(\theta_\star+\delta)
\approx
\hat R_n(\theta_\star)+\frac{1}{2}\operatorname{tr}(H\Sigma).
$$
So a flat solution can tolerate a broader posterior without much empirical loss increase, while a broader posterior may reduce the KL penalty. This is the intuition behind neural-network PAC-Bayes analyses such as [Dziugaite and Roy 2017](https://arxiv.org/abs/1703.11008) and perturbation- or margin-based bounds such as [Neyshabur, Bhojanapalli, McAllester, and Srebro 2018](https://openreview.net/forum?id=Skz_WfbCZ).

The same lesson appears in norm- and margin-based bounds such as [Bartlett, Foster, and Telgarsky 2017](https://proceedings.neurips.cc/paper/2017/hash/b22b257ad0519d4500539da3c8bcf4dd-Abstract.html): what matters is not raw parameter count, but scale-sensitive control of sensitivity and margin. Flatness is therefore best understood not as a standalone magic variable, but as one way of quantifying perturbation stability when the geometry is chosen carefully.

## 6. Implicit Low-Dimensional Effects

Deep networks are often trained in parameter spaces whose nominal dimension is much larger than the number of directions that seem to matter in practice. One clue comes from Hessian spectra: a small number of dominant eigenvalues often account for most local curvature, while many directions are nearly flat. This suggests that optimization and local generalization may depend on an active subspace much smaller than the raw parameter count.

[Li et al. 2018](https://openreview.net/forum?id=ryup8-WCW) studied this more directly through intrinsic dimension experiments, showing that some tasks can be solved even when optimization is restricted to a random low-dimensional subspace. [Gur-Ari, Roberts, and Dyer 2019](https://openreview.net/forum?id=ByeTHsAqtX) argued that gradient descent quickly aligns with a small top-Hessian subspace. These findings do not imply that the model really has only a few parameters, but they do suggest that effective dimension can be much smaller than nominal dimension.

This perspective becomes even clearer in modern fine-tuning. [Aghajanyan, Gupta, and Zettlemoyer 2021](https://aclanthology.org/2021.acl-long.568/) argued that the intrinsic dimension needed for language-model adaptation can be quite small, and [Hu et al. 2022](https://openreview.net/forum?id=nZeVKeeFYf9) showed with LoRA that low-rank updates often suffice for large-model adaptation. Here the base representation learned during pretraining carries much of the complexity, while downstream learning happens in a restricted subspace.

Pruning, sparsity, and compression support the same picture from another angle. Classical second-order pruning methods such as Optimal Brain Damage and Optimal Brain Surgeon use curvature to identify dispensable directions. Modern compression work such as [Han, Mao, and Dally 2016](https://arxiv.org/abs/1510.00149) shows that many trained networks can be represented much more compactly than their dense parameterization suggests, while [Arora, Ge, Neyshabur, and Zhang 2018](https://proceedings.mlr.press/v80/arora18b.html) connect compressibility to generalization bounds. The [lottery ticket hypothesis](https://openreview.net/forum?id=rJl-b3RcF7) of Frankle and Carbin 2019 pushes this further by suggesting that a much smaller performant subnetwork may already exist inside the dense model.

The important caveat is that low-dimensional parameter movement is not the same as low function complexity. A low-rank update can still implement a complicated function if the pretrained representation is rich. So implicit low-dimensional effects are best interpreted as statements about optimization, redundancy, and representation structure, not as a universal replacement for statistical complexity.

## 7. More on the Shape of the Loss Landscape

Flatness describes local curvature near one solution, but the global loss landscape contains more structure than one Hessian can capture. Overparameterized networks often appear to have wide connected regions of low loss rather than isolated minima. This changes the intuitive picture of optimization: training may be entering a low-loss manifold or valley system rather than locating a single distinguished point.

Mode connectivity is the most striking example. [Garipov et al. 2018](https://papers.neurips.cc/paper/8095-loss-surfaces-mode-connectivity-and-fast-ensembling-of-dnns) and [Draxler et al. 2018](https://proceedings.mlr.press/v80/draxler18a.html) found low-loss paths connecting independently trained solutions. This suggests that many apparent minima lie in a larger connected structure. Such geometry helps explain why weight averaging and ensembling methods can work so well: if many trained models lie in one broad valley, averaging can remain in a low-loss region instead of falling into a barrier.

Symmetry is a major reason the global geometry is nontrivial. Hidden units can be permuted without changing the represented function, and positively homogeneous layers can be rescaled across adjacent layers. Some apparent barriers in parameter space are therefore not barriers in function space. Work such as [Ainsworth, Hayase, and Srinivasa 2023](https://openreview.net/forum?id=CQsmMYmlP5T) shows that after accounting for permutation symmetry, solutions can become much more directly connected.

There are also more global theoretical hints that overparameterization changes landscape topology. Deep linear networks and some overparameterized nonlinear models have no bad local minima under suitable assumptions, and some sublevel sets become connected or nearly connected. These results do not solve realistic deep learning, but they support the broader lesson of this section: local curvature is only one piece of landscape geometry, and global connectivity, symmetry, barriers, and manifold structure also matter for how optimization and generalization interact.

## 8. Robustness, Shift, and Caveats

Generalization in the narrow sense usually means low test error on data drawn from the same distribution as the training set. Robustness, calibration, and distribution-shift performance are different questions. A model can generalize well on clean i.i.d. test data and still be brittle under adversarial perturbations, corruptions, demographic shifts, or task transfer. Weight-space flatness alone does not guarantee input-space robustness.

This is one reason flatness and related measures must be interpreted carefully. Raw sharpness can change under reparameterization without changing the function, PAC-Bayes and Laplace arguments are often local, and empirical correlations between complexity measures and test error can disappear when the experimental protocol changes. [Jiang et al. 2020](https://openreview.net/forum?id=SJgIPJBFvH) is especially important here because it shows that many proposed generalization measures are not uniformly reliable across architectures and training setups.

A careful survey should therefore distinguish three kinds of statements. A correlation claim says that a measure tracks test error in some experiments. A bound says that a measure upper-bounds risk under explicit assumptions. A mechanism claim says that changing the measure causally changes generalization. These are not the same level of explanation. Many results in deep learning are best read as regime-dependent evidence rather than universal causal laws.

The right conclusion is not that the topic is hopeless, but that deep-learning generalization is multi-mechanism. Norms and margins matter, algorithmic stability matters, optimizer bias matters, Bayesian volume matters, flatness sometimes matters, and low-dimensional or compressible structure often matters. The open problem is to identify, for each architecture-data-training regime, which of these mechanisms is actually doing the work.
