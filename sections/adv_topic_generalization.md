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

1. classical TODO:

1. What changed in deep learning

    more complicated loss landscape


    Classical generalization theory often controls a whole hypothesis class uniformly, using VC dimension, Rademacher complexity, covering numbers, margins, or norms. Modern deep networks complicate this picture because they are heavily overparameterized, can drive training error to zero, and yet often generalize well. The survey therefore separates capacity of the architecture from the implicit restrictions imposed by data, optimization, initialization, normalization, and training protocol.

2. Experimental phenomenology

    The starting point is empirical. Networks can fit random labels, small-batch and large-batch training may converge to different solutions, learning-rate schedules and batch size affect sharpness, and different optimizers can select different interpolating models. These observations motivate flatness, implicit regularization, and loss-landscape explanations, but they also show why a single scalar complexity measure is unlikely to explain every regime.

3. Measuring flatness and curvature

    A flat minimum is a solution whose loss changes slowly under parameter perturbations. In practice, people measure top Hessian eigenvalues, Hessian trace, spectral density, perturbation sharpness, local entropy, normalized sharpness, or filter-normalized visualizations. These quantities are related through the quadratic approximation, but they are not identical, and raw parameter-space sharpness is not invariant under common neural-network reparameterizations.

4. Theoretical results: stationary phase, Laplace evidence, and the Hessian

    The cleanest mathematical route from curvature to model complexity is the asymptotic expansion of an integral over parameters. Around a regular nondegenerate minimum, stationary phase or Laplace's method gives an Occam factor involving $\det H$, where $H$ is the Hessian of the empirical loss. This explains why wide basins can have higher Bayesian evidence, but it also exposes a caveat: deep networks are often singular and degenerate, so ordinary Hessian-based Laplace theory must be used locally and carefully.

5. PAC-Bayes, perturbation stability, and norm-based bounds

    A second theoretical route is to bound the risk of a distribution over parameters. If perturbing a trained model does not increase empirical loss much, and if the perturbation distribution remains close to a prior, PAC-Bayes gives a nonvacuous path from flatness to generalization. Related norm and margin bounds replace raw parameter count by scale-sensitive quantities such as spectral norms, path norms, margins, and compression length.

6. Implicit regularization of optimization

    Overparameterized training has many empirical minimizers, so the algorithm matters. Gradient descent, stochastic gradient descent, Adam, weight decay, normalization, sharpness-aware minimization, stochastic weight averaging, and learning-rate schedules all bias the selected solution. In simple models this bias can be characterized exactly, for example as max-margin behavior; in deep networks the theory is less complete but still central.

7. Implicit low-dimensional effects

    Deep learning often behaves as if training were happening in a much smaller effective space than the raw parameter dimension. Hessian spectra have many near-zero directions and a small number of outliers; random low-dimensional subspaces can sometimes support successful training; fine-tuning large language models often needs only low-rank or low-dimensional updates. These observations connect flatness, representation learning, pruning, and parameter-efficient fine-tuning.

8. Pruning, sparsity, and compression

    Pruning asks whether a trained network contains a smaller subnetwork that preserves performance. Compression-based generalization bounds say that a predictor that can be described with fewer bits should generalize better, while lottery-ticket experiments show that sparse subnetworks can sometimes train or transfer surprisingly well. Hessian-aware pruning makes the connection to curvature explicit by removing weights that least increase the quadratic loss.

9. Shape of the loss landscape

    Local curvature is only one part of landscape geometry. The global landscape includes mode connectivity, low-loss paths between independently trained models, low-loss manifolds, permutation symmetries, barriers, and the geometry induced by normalization and residual connections. This section studies the shape of the loss landscape as a global object rather than reducing it to one Hessian at one solution.

10. Neural tangent kernel and lazy training
    
    In the infinite-width neural tangent kernel regime, the network behaves like a kernel method around initialization. The tangent kernel stays almost fixed, training is approximately linear in parameters, and generalization is governed by the data geometry and the induced kernel rather than by feature learning. This gives a rigorous theory of some overparameterized networks, but it is not the whole story.

11. Mean-field theory and feature learning

    Mean-field theory studies a different infinite-width limit in which the distribution of neurons evolves during training. This captures feature learning more directly than the lazy NTK regime and leads to gradient-flow equations over probability measures. The useful contrast is not "NTK versus mean field" as competing slogans, but kernel-like interpolation versus representation-changing dynamics.

12. Negative results and survey methodology

    Generalization measures must be interpreted with care. Raw flatness can be changed by reparameterization without changing the represented function; many measures correlate with test error only under restricted experimental protocols; and robustness, calibration, and distribution-shift performance are not automatic consequences of clean test generalization. A good survey records the architecture, optimizer, batch size, learning rate, normalization convention, perturbation radius, and whether the paper claims correlation, bound, or causal mechanism.

## 1. What Generalization Asks

The basic quantity is the generalization gap
$$
R(\theta)-\hat R_n(\theta).
$$
Classical learning theory often controls this gap uniformly over a class $\mathcal{F}$:
$$
\sup_{f\in\mathcal{F}} |R(f)-\hat R_n(f)|.
$$
This gives clean results when the effective size of $\mathcal{F}$ is small relative to $n$. For example, VC-dimension and Rademacher-complexity bounds say that large classes require more data unless they are restricted by margins, norms, sparsity, stability, or other structure. The difficulty for deep learning is that the raw architecture class is usually enormous. A modern network can have far more parameters than training examples and can still generalize well on natural data.

The empirical challenge was made especially vivid by [Zhang, Bengio, Hardt, Recht, and Vinyals 2017](https://research.google/pubs/understanding-deep-learning-requires-rethinking-generalization/). They showed that standard deep networks can fit random labels with zero training error, while the same architectures generalize on real labels. This means that architecture-level capacity alone cannot be the full explanation. The trained predictor is not merely a random element of the network class; it is the result of a data-dependent algorithmic process.

There are therefore several different questions that should not be conflated. Expressiveness asks whether the model class can represent a function. Optimization asks whether training can find low empirical loss. Generalization asks whether the found function has low population risk. In overparameterized networks, all three interact: high expressiveness permits interpolation; optimization chooses one interpolating solution among many; and generalization depends on which solution is selected and how it relates to the data distribution.

The most useful survey stance is conditional rather than universal. A theory may explain generalization in one regime, such as kernel-like training near initialization, max-margin classification, compressed sparse networks, smooth data manifolds, or Bayesian local evidence around a minimum. It need not explain every deep network in every training setup. The important task is to identify the assumptions under which a mechanism is actually active.

## 2. Experimental Phenomenology

The flat-minima story begins from the observation that different training protocols can reach solutions with similar training error but different test error. [Hochreiter and Schmidhuber 1997](https://direct.mit.edu/neco/article/9/1/1/6027/Flat-Minima) argued that flat minima should generalize better because many nearby parameter settings implement similarly good predictors, while sharp minima are more sensitive to perturbations. [Keskar, Mudigere, Nocedal, Smelyanskiy, and Tang 2017](https://openreview.net/forum?id=H1oyRlYgg) connected this idea to the generalization gap observed in some large-batch training runs: large batches can reduce gradient noise and may converge to sharper solutions than small-batch training under otherwise similar hyperparameters.

The empirical picture is more nuanced than "small batch good, large batch bad." Large-batch training can generalize well when learning-rate schedules, warmup, regularization, and training time are adjusted. [Hoffer, Hubara, and Soudry 2017](https://papers.nips.cc/paper/6770-train-longer-generalize-better-closing-the-generalization-gap-in-large-batch-training-of-neural-networks) emphasized that part of the large-batch gap can come from comparing different numbers of parameter updates rather than only different batch sizes. The broader lesson is that batch size affects optimization noise, but the outcome depends on the full training protocol.

Several algorithms deliberately bias training toward flatter or wider solutions. [Chaudhari et al. 2017](https://openreview.net/forum?id=B1YfAfcgl) proposed Entropy-SGD, which optimizes a local-entropy objective that rewards neighborhoods of low loss. [Izmailov, Podoprikhin, Garipov, Vetrov, and Wilson 2018](https://proceedings.mlr.press/v80/izmailov18a.html) introduced stochastic weight averaging, which averages weights along the SGD trajectory and often lands in wider basins with improved test performance. [Foret, Kleiner, Mobahi, and Neyshabur 2021](https://openreview.net/forum?id=6Tm1mposlrM) proposed sharpness-aware minimization (SAM), which minimizes the worst-case loss in a local parameter neighborhood and has become one of the main practical examples of flatness-inspired training.

The experimental literature also shows that generalization is not controlled by flatness alone. Data augmentation, label smoothing, weight decay, dropout, normalization layers, residual connections, pretraining, architecture, and optimizer all change the selected solution. [Jiang et al. 2020](https://openreview.net/forum?id=SJgIPJBFvH) evaluated many proposed generalization measures and found that their predictive power depends strongly on experimental context. This is why later sections treat flatness as one mechanism among several, not as a complete theory.

## 3. Curvature, Flatness, and Measurement

Let $\theta_\star$ be a trained solution and let
$$
H=\nabla^2 \hat R_n(\theta_\star)
$$
be the Hessian of the empirical risk. If $\nabla \hat R_n(\theta_\star)\approx 0$, then for a small perturbation $\delta$ the Taylor expansion gives
$$
\hat R_n(\theta_\star+\delta)-\hat R_n(\theta_\star)\approx \frac{1}{2}\delta^\top H\delta.
$$
This formula explains several common flatness measures. The worst-case quadratic loss increase over a Euclidean ball of radius $\epsilon$ is approximately $\frac{1}{2}\epsilon^2\lambda_{\max}(H)$. The expected increase under an isotropic Gaussian perturbation $\delta\sim \mathcal{N}(0,\sigma^2 I)$ is approximately $\frac{\sigma^2}{2}\operatorname{tr}(H)$. A volume or Bayesian-evidence calculation involves $\log\det H$ or a pseudo-determinant after removing nearly flat directions.

The Hessian spectrum of trained networks is highly structured. Empirical studies such as [Sagun, Evci, Guney, Dauphin, and Bottou 2017](https://arxiv.org/abs/1706.04454) and [Ghorbani, Krishnan, and Xiao 2019](https://proceedings.mlr.press/v97/ghorbani19b.html) find a small number of large outlier eigenvalues and a large bulk of small eigenvalues. This supports the idea that many parameter directions are nearly flat while a few directions control most local curvature. It also means that reporting only one scalar, such as the largest eigenvalue, can miss much of the geometry.

Visualizations are useful but fragile. [Li et al. 2018](https://proceedings.neurips.cc/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html) showed that naive random 2D slices through weight space can be misleading and introduced filter normalization to make loss-landscape plots more comparable across networks. These plots are best read as qualitative diagnostics. They can reveal barriers, valleys, and basin geometry, but they do not replace invariant quantitative definitions.

The main caveat is reparameterization. [Dinh, Pascanu, Bengio, and Bengio 2017](https://proceedings.mlr.press/v70/dinh17b.html) showed that, because ReLU networks have positive-homogeneity rescalings, one can make a solution arbitrarily sharp in raw parameter coordinates without changing the represented function. Batch normalization and other scale symmetries create similar issues. Therefore a useful flatness measure must specify the parameterization, normalization convention, perturbation geometry, and scale. Alternatives such as normalized sharpness, adaptive sharpness, margin-normalized measures, path norms, Fisher geometry, and function-space perturbations are attempts to repair this problem.

## 4. Stationary Phase, Laplace Evidence, and PAC-Bayes

The most classical link between flatness and generalization comes from Bayesian model comparison. Consider a posterior or evidence integral of the form
$$
Z_n(\beta)=\int_\Theta \exp(-\beta n\hat R_n(\theta))\pi(\theta)\,d\theta,
$$
where $\pi$ is a prior and $\beta$ is an inverse-temperature parameter. Suppose the empirical risk has isolated nondegenerate minima $\theta_1,\ldots,\theta_k$ with positive-definite Hessians $H_j=\nabla^2\hat R_n(\theta_j)$. Expanding $\hat R_n$ to second order around each minimum and applying stationary phase, or equivalently Laplace's method for a positive real exponential, gives
$$
Z_n(\beta)\approx \sum_{j=1}^k \exp(-\beta n\hat R_n(\theta_j))\pi(\theta_j)\left(\frac{2\pi}{\beta n}\right)^{p/2}\frac{1}{\sqrt{\det H_j}},
$$
where $p$ is the parameter dimension. Taking negative logarithms shows the Occam factor:
$$
-\log Z_n(\beta)\approx \beta n\hat R_n(\theta_j)+\frac{1}{2}\log\det H_j+\frac{p}{2}\log\frac{\beta n}{2\pi}-\log\pi(\theta_j).
$$
For two minima with the same training loss and prior density, the one with smaller $\det H$ occupies more posterior volume and contributes more to the evidence. This is the clean mathematical version of "flat minima are favored."

This Hessian calculation appears in classical Bayesian neural-network and minimum-description-length work, including [MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.448) and [Hinton and van Camp 1993](https://dl.acm.org/doi/10.1145/168304.168306). It is important because it derives the Hessian rather than assuming it: curvature appears as the Gaussian normalization constant in the local asymptotic integral. The result also clarifies which notion of flatness is being used: the determinant of the local quadratic form, not merely the top eigenvalue.

Deep networks violate the regular nondegenerate assumptions in many ways. They have permutation symmetries, scale symmetries, redundant parameters, rank-deficient Hessians, and singular parameterizations in which many parameter values represent the same function. In such cases the ordinary Laplace approximation can fail or must be modified by removing zero modes and accounting for singular geometry. [Watanabe 2013](https://www.jmlr.org/papers/v14/watanabe13a.html) develops the widely applicable Bayesian information criterion for singular models, where the asymptotic penalty is governed by the real log canonical threshold rather than the regular-model dimension $p/2$. The lesson is not that Hessian arguments are useless; it is that they are local regular approximations unless the singular structure is handled explicitly.

PAC-Bayes gives another route from flatness to generalization. Instead of studying a single parameter vector, choose a posterior distribution $Q$ over parameters, often centered near the trained solution, and compare it to a prior $P$. A typical PAC-Bayes theorem bounds the population risk averaged over $Q$ by an empirical risk term plus a complexity term depending on $\mathrm{KL}(Q\|P)$ and $n$. If $Q=\mathcal{N}(\theta_\star,\Sigma)$, then a local expansion gives
$$
\mathbb{E}_{\delta\sim\mathcal{N}(0,\Sigma)}\hat R_n(\theta_\star+\delta)\approx \hat R_n(\theta_\star)+\frac{1}{2}\operatorname{tr}(H\Sigma).
$$
Thus a flat solution can tolerate a broader posterior $Q$ without increasing empirical loss, while a broader posterior can reduce the KL penalty if it remains compatible with the prior.

This idea underlies nonvacuous and nearly nonvacuous bounds for neural networks such as [Dziugaite and Roy 2017](https://arxiv.org/abs/1703.11008), margin- and perturbation-based analyses such as [Neyshabur, Bhojanapalli, McAllester, and Srebro 2018](https://openreview.net/forum?id=Skz_WfbCZ), and norm-based margin bounds such as [Bartlett, Foster, and Telgarsky 2017](https://proceedings.neurips.cc/paper/2017/hash/b22b257ad0519d4500539da3c8bcf4dd-Abstract.html). The common synthesis is that a useful bound must control both empirical fit and sensitivity: low training loss alone is not enough, and flatness alone is not enough unless it is measured in a scale-aware geometry.

## 5. Classical Capacity, Stability, and Double Descent

Before turning to implicit regularization, it is worth separating three classical theoretical families. First, capacity bounds control a hypothesis class uniformly. VC dimension, pseudodimension, covering numbers, Rademacher complexity, and margin bounds all fit this template. For neural networks, parameter-count bounds are usually too loose in overparameterized regimes, but norm- and margin-sensitive versions can still give useful qualitative guidance: large margins and controlled norms reduce effective complexity.

Second, stability bounds control the sensitivity of the training algorithm to changing one example. [Hardt, Recht, and Singer 2016](https://proceedings.mlr.press/v48/hardt16.html) prove algorithmic-stability bounds for stochastic gradient descent under convex smooth assumptions and show how step sizes and training time enter generalization. The deep nonconvex case is harder, but the conceptual point remains valuable: generalization can be a property of the algorithm, not only of the final hypothesis class.

Third, interpolation itself is not fatal. The double-descent literature studies how test error can decrease again after a model crosses the interpolation threshold. [Belkin, Hsu, Ma, and Mandal 2019](https://www.pnas.org/doi/10.1073/pnas.1903070116) argued that classical bias-variance intuition should be extended to overparameterized interpolating models, and [Nakkiran et al. 2021](https://openreview.net/forum?id=B1g5sA4twr) demonstrated deep double descent across model size, epoch count, and sample size. This does not explain every deep-learning success, but it removes the simplistic assumption that interpolation necessarily implies poor generalization.

The key synthesis is that overparameterization changes which complexity measure matters. The relevant quantity is often not the number of parameters, but the norm, margin, stability, compression length, kernel complexity, feature dimension, or local volume of the solution selected by training.

## 6. Implicit Regularization of Optimization

In an overparameterized model, many parameters can satisfy $\hat R_n(\theta)=0$. Training therefore acts as an implicit regularizer by selecting one solution from a large set. In linear models, this selection can be made precise: gradient descent on squared loss from small initialization converges to the minimum Euclidean-norm interpolant, and gradient descent on separable logistic regression converges in direction to the max-margin classifier, as shown by [Soudry, Hoffer, Nacson, Gunasekar, and Srebro 2018](https://jmlr.org/papers/v19/18-188.html). For matrix factorization and homogeneous models, related work such as [Gunasekar et al. 2018](https://proceedings.mlr.press/v80/gunasekar18a.html) shows biases toward low-rank or norm-controlled solutions.

For deep networks, implicit regularization is harder to characterize because the parameterization is nonlinear, homogeneous symmetries matter, normalization changes the effective geometry, and the loss landscape has many connected regions. Still, several recurring mechanisms appear. Small initialization can bias training toward low-complexity functions early in training. Weight decay interacts with scale-invariant layers by affecting effective learning rates. Batch normalization changes both optimization geometry and the meaning of weight norms. Data augmentation changes the effective training distribution and can be interpreted as enforcing invariances.

Stochasticity is another major ingredient. SGD noise depends on batch size, learning rate, data order, and gradient covariance. [Mandt, Hoffman, and Blei 2017](https://jmlr.org/papers/v18/17-214.html) modeled SGD as approximate Bayesian inference near a local optimum, while [Smith and Le 2018](https://openreview.net/forum?id=BJij4yg0Z) argued for a Bayesian perspective on the learning-rate/batch-size ratio. These models are approximations, but they explain why optimization noise can favor broad basins and why changing batch size without changing learning rate can change the selected solution.

Flatness-oriented optimizers make the selection effect explicit. SAM solves an inner maximization over a local neighborhood before taking a descent step, approximately optimizing
$$
\min_\theta \max_{\|\epsilon\|\leq \rho} \hat R_n(\theta+\epsilon).
$$
In the small-$\rho$ regime, the inner problem is connected to gradient norm and curvature; in practice, SAM often improves clean test accuracy and robustness to some distributional changes, but it is still sensitive to the perturbation geometry and normalization. The main conceptual point is that optimization is not merely a way to minimize training loss. It defines which interpolating solution is preferred.

## 7. Implicit Low-Dimensional Effects

Although deep networks may contain millions or billions of parameters, several empirical results suggest that the effective degrees of freedom used by training can be much smaller. [Li et al. 2018](https://openreview.net/forum?id=ryup8-WCW) measured the intrinsic dimension of objective landscapes by training networks in random low-dimensional parameter subspaces and found that successful optimization can occur in dimensions far below the full parameter count for some tasks. This does not mean the model has only that many meaningful parameters, but it shows that the raw dimension can badly overstate the dimension needed for successful search.

Hessian studies give a complementary view. If the Hessian has a small number of large eigenvalues and many near-zero directions, then most local parameter perturbations have little effect on training loss. [Gur-Ari, Roberts, and Dyer 2019](https://openreview.net/forum?id=ByeTHsAqtX) argued that gradient descent quickly enters a small "active" subspace associated with the top Hessian eigenspace, after which most learning happens in that subspace. This connects flatness to low-dimensional training dynamics: a model can be huge, but the local curvature and gradients may live in a much smaller effective space.

Fine-tuning large pretrained models strengthens this theme. [Aghajanyan, Gupta, and Zettlemoyer 2021](https://aclanthology.org/2021.acl-long.568/) argued that the intrinsic dimension needed for language-model fine-tuning is much smaller than the full parameter count and decreases with pretraining quality. [Hu et al. 2022](https://openreview.net/forum?id=nZeVKeeFYf9) proposed LoRA, which freezes pretrained weights and learns low-rank updates, showing that many adaptation tasks can be handled by a low-rank subspace. These results are not generalization proofs by themselves, but they explain why low-dimensional structure is a plausible part of modern generalization, especially in transfer learning.

The caveat is that parameter-space dimension and function-space complexity are different. A low-dimensional update can still implement a complex function if the pretrained representation is rich. Conversely, a high-dimensional parameter vector can represent a simple function if many parameters are redundant. The low-dimensional effect is therefore best understood as an interaction between data, initialization, representation, and optimization, not as a universal replacement for statistical complexity.

## 8. Pruning, Sparsity, and Compression

Pruning is one of the clearest empirical signs of redundancy in neural networks. Classical second-order pruning methods such as [Optimal Brain Damage](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) by LeCun, Denker, and Solla and [Optimal Brain Surgeon](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html) by Hassibi and Stork used Hessian information to estimate which weights could be removed with minimal increase in training loss. In the quadratic approximation, removing or perturbing parameters is cheap along flat directions and expensive along sharp directions.

Modern compression work broadened the picture. [Han, Mao, and Dally 2016](https://arxiv.org/abs/1510.00149) showed that pruning, quantization, and coding can dramatically reduce deployed network size. From a theory perspective, compression suggests a generalization principle: if a trained network can be encoded with few bits while preserving training performance, then the effective hypothesis class of compressed predictors is smaller. [Arora, Ge, Neyshabur, and Zhang 2018](https://proceedings.mlr.press/v80/arora18b.html) developed compression-based generalization bounds along these lines.

The lottery-ticket hypothesis of [Frankle and Carbin 2019](https://openreview.net/forum?id=rJl-b3RcF7) made pruning more structural. It proposed that dense randomly initialized networks contain sparse subnetworks that, when trained in isolation from suitable initial weights, can match the full network's performance. Later work refined when and how such tickets appear, but the central message remains useful for generalization: overparameterization may make optimization easier while the final function can sometimes be represented by a much smaller subnetwork.

Compression is not automatically a causal explanation. A network can be compressible after training because the learned function is simple, because the data distribution is simple, because pruning exploits redundancy created by overparameterized optimization, or because the compression procedure itself performs additional training. Still, pruning and compression provide concrete bridges among flatness, low dimension, sparsity, and minimum-description-length intuition.

## 9. Shape of the Loss Landscape

The local Hessian describes a small neighborhood around one solution, but the global loss landscape can be much richer. Early theory on simplified deep networks showed that some idealized landscapes have no bad local minima. For example, [Kawaguchi 2016](https://papers.nips.cc/paper/6112-deep-learning-without-poor-local-minima) analyzed deep linear networks and showed that all local minima are global under suitable assumptions, while [Freeman and Bruna 2017](https://openreview.net/forum?id=S1thXcEkM) studied connectedness properties of sublevel sets in overparameterized ReLU networks. These results do not directly prove that realistic networks are easy to train, but they show how overparameterization can change landscape topology.

Mode connectivity is one of the most striking empirical landscape findings. [Garipov et al. 2018](https://papers.neurips.cc/paper/8095-loss-surfaces-mode-connectivity-and-fast-ensembling-of-dnns) and [Draxler et al. 2018](https://proceedings.mlr.press/v80/draxler18a.html) found that independently trained neural networks can often be connected by simple low-loss curves in parameter space. This weakens the picture of isolated basins separated by high barriers. A "minimum" in a modern network is often better thought of as part of a broad connected region or manifold of low loss.

Symmetry complicates global geometry. Hidden units can often be permuted without changing the represented function, and homogeneous layers can be rescaled across adjacent layers. Recent work on linear mode connectivity after weight matching, such as [Ainsworth, Hayase, and Srinivasa 2023](https://openreview.net/forum?id=CQsmMYmlP5T), shows that some apparent barriers are artifacts of permutation symmetries rather than intrinsic functional differences. This reinforces the earlier flatness caveat: parameter-space geometry is informative only after accounting for the representation's symmetries.

The practical consequence is that landscape geometry can support generalization in several ways. Wide connected regions make stochastic weight averaging and ensembling more effective. Low-loss valleys can contain many predictors with similar training loss but different test behavior. Flat directions can permit pruning and fine-tuning. But landscape geometry alone does not determine generalization; the same landscape must be interpreted together with data geometry and optimization dynamics.

## 10. Neural Tangent Kernel and Lazy Training

The neural tangent kernel gives a rigorous theory for one important overparameterized regime. Let $f_\theta(x)$ be a network and define the tangent kernel
$$
K_\theta(x,x')=\nabla_\theta f_\theta(x)^\top \nabla_\theta f_\theta(x').
$$
Near initialization, the first-order expansion is
$$
f_\theta(x)\approx f_{\theta_0}(x)+\nabla_\theta f_{\theta_0}(x)^\top(\theta-\theta_0).
$$
If the network is sufficiently wide under the standard NTK scaling, the tangent kernel changes little during training. Gradient descent in parameter space then becomes approximately kernel gradient descent in function space.

[Jacot, Gabriel, and Hongler 2018](https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html) introduced the NTK limit, and [Lee et al. 2019](https://papers.nips.cc/paper/9063-wide-neural-networks-of-any-depth-evolve-as-linear-models-under-gradient-descent) showed that wide networks of any depth can evolve as linear models under gradient descent in this regime. Generalization is then controlled by the induced kernel, the target function's alignment with the kernel eigenfunctions, the noise level, regularization, and the sample size. The theory explains how an infinitely overparameterized model can interpolate without arbitrary behavior: the implicit bias is kernel regression with a specific data-dependent kernel.

The NTK regime is powerful because it is mathematically tractable. It gives global convergence results, exact training dynamics in the infinite-width limit, and connections to classical kernel generalization. It also helps explain why overparameterization can make optimization easier: the linearized model has a well-conditioned tangent feature representation when the kernel matrix is favorable.

The limitation is equally important. In lazy training, features barely change, so representation learning is mostly absent. Many practical networks, especially those trained with finite width, large learning rates, normalization, pretraining, or feature-rich architectures, appear to learn representations that improve over the initial kernel. Therefore NTK theory is a correct theory of a regime, not a complete theory of deep learning.

## 11. Mean-Field Theory and Feature Learning

Mean-field theory studies a different infinite-width limit. For a two-layer network, write
$$
f_\rho(x)=\int a\,\sigma(w^\top x)\,d\rho(a,w),
$$
where $\rho$ is a distribution over neurons. Training becomes the evolution of $\rho$ rather than the movement of finitely many individual neurons. In the infinite-width limit, gradient descent can converge to a gradient flow over probability measures:
$$
\partial_t \rho_t = \nabla\cdot\left(\rho_t \nabla \frac{\delta R(\rho_t)}{\delta \rho}\right),
$$
with the exact form depending on the parameterization and loss.

This line includes [Mei, Montanari, and Nguyen 2018](https://doi.org/10.1073/pnas.1806579115), [Chizat and Bach 2018](https://proceedings.neurips.cc/paper/2018/hash/a1afc58c6ca9540d057299ec3016d726-Abstract.html), [Rotskoff and Vanden-Eijnden 2018](https://arxiv.org/abs/1805.00915), and [Sirignano and Spiliopoulos 2020](https://doi.org/10.1137/18M1192184). These works make different assumptions, but they share the idea that an overparameterized network can be treated as an interacting-particle system or probability distribution whose evolution changes the feature representation.

The contrast with NTK was sharpened by [Chizat, Oyallon, and Bach 2019](https://proceedings.neurips.cc/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html), who described lazy training as a regime caused by parameterization and scaling: if the network output is scaled so that parameters move little, the model behaves like a kernel method; if parameters move substantially, the model can enter a richer feature-learning regime. [Woodworth et al. 2020](https://proceedings.mlr.press/v125/woodworth20a.html) studied kernel and rich regimes in overparameterized models and showed that the same architecture can have different implicit biases depending on scaling and optimization.

Mean-field theory is attractive for generalization because it gives a language for learned features, not only fixed kernels. It can explain how a network adapts its representation to data, why width helps optimization, and how distributional dynamics can avoid some finite-particle pathologies. But the theory is still most mature for simplified two-layer or special architectures. Extending it cleanly to practical transformers, normalization-heavy networks, and modern pretraining remains an active research direction.

## 12. Robustness, Shift, and Caveats

Generalization is often used narrowly to mean clean test error on the same distribution as the training data. Robustness and distribution shift are different questions. A solution that is flat under small parameter perturbations need not be robust to adversarial input perturbations, corruptions, demographic shifts, or task shifts. Some flatness-oriented methods such as SAM can improve robustness to certain shifts, but there is no theorem saying that weight-space flatness implies input-space robustness in general.

Uncertainty and calibration also require separate treatment. Bayesian Laplace methods and posterior approximations use curvature to estimate uncertainty around a trained solution, but a local Gaussian approximation can be poor in singular or multimodal neural-network posteriors. Ensembles, SWA-Gaussian methods, and Laplace approximations can improve uncertainty estimates in some settings, yet they depend on which modes or valleys are explored.

Negative results are therefore essential rather than incidental. Raw sharpness can be changed by reparameterization without changing the predictor, as shown by [Dinh et al. 2017](https://proceedings.mlr.press/v70/dinh17b.html). Many generalization measures work only after controlling for architecture, optimizer, data augmentation, and hyperparameters, as emphasized by [Jiang et al. 2020](https://openreview.net/forum?id=SJgIPJBFvH). Kernel-limit theories can miss feature learning; mean-field theories can rely on idealized width limits; pruning can describe a trained network without explaining why training found it; and Bayesian local evidence can fail in singular settings.

The right conclusion is not pessimistic. The literature has identified several real mechanisms: margins and norms, algorithmic stability, Bayesian volume, perturbation robustness, optimizer-induced bias, low-dimensional structure, compression, connected low-loss landscapes, kernel interpolation, and feature-learning dynamics. The open problem is to know which mechanism dominates in a given architecture-data-training regime.

## Takeaway

The generalization of deep learning is not explained by a single number such as parameter count, Hessian top eigenvalue, or training loss. The modern view is multi-mechanism. Overparameterized networks can fit arbitrary labels, but real training on real data selects special solutions through initialization, optimization, architecture, normalization, augmentation, and pretraining. Flatness is important when it is measured in an appropriate geometry and connected to perturbation stability, Bayesian volume, or compression. NTK theory gives a rigorous kernel-like regime; mean-field theory describes representation-changing regimes; pruning and low-dimensional fine-tuning show that the effective degrees of freedom can be far smaller than the raw parameter count. A careful survey should therefore state the regime first and only then state the generalization mechanism.
