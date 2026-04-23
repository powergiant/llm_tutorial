# Convergence of Deep Learning

## Outline

In the expressiveness section, the central question was whether the chosen architecture can represent or approximate the target function. Convergence asks the next question: after we choose a model family and a training objective, does the optimization procedure actually move the parameters toward a useful solution, and what kind of solution does it select?

The basic supervised objective is
$$
L_n(\theta)=\frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x_i),y_i),
$$
and a very broad template for modern training is
$$
\theta_{t+1}=\theta_t-\eta_t P_t g_t,
$$
where $g_t$ is either the full gradient $\nabla L_n(\theta_t)$ or a stochastic mini-batch gradient, $\eta_t$ is the learning rate, and $P_t$ is an optimizer-dependent transformation such as momentum, diagonal adaptivity, or matrix preconditioning. This section surveys what is known about convergence for this update, from classical convex theory to the large-learning-rate and large-model phenomena that appear in deep learning.

1. **Notions of convergence**

   Convergence can mean objective decrease, small gradient norm, approach to a local or global minimizer, convergence of the represented function $f_{\theta_t}$, convergence of learned representations, or improvement of validation loss and downstream metrics. These are not equivalent in deep learning because the parameterization is non-identifiable, the loss is nonconvex, and training often continues long after the training error is zero.

2. **Empirical convergence phenomena**

   Before theory, we should describe what practitioners measure: training loss, validation loss, gradient norm, update norm, sharpness, Hessian spectrum, batch-size dependence, seed sensitivity, wall-clock efficiency, and token efficiency. Important empirical facts include fast interpolation, continued improvement after interpolation, large-batch scaling limits, loss spikes, schedule-dependent phase transitions, and the difference between step convergence and compute convergence.

3. **Convex optimization background**

   Convex optimization gives the clean reference theory. Gradient descent on smooth convex objectives has $O(1/T)$ objective convergence, strongly convex objectives have linear convergence, stochastic gradients give slower expectation bounds, and Polyak-Lojasiewicz objectives can have linear convergence without full convexity. These results are not directly deep-learning results, but they define the baseline assumptions that deep learning often violates.

4. **Nonconvex optimization background**

   Generic smooth nonconvex theory usually guarantees convergence only to first-order stationarity, measured by $\|\nabla L(\theta)\|$. Stronger second-order guarantees require escaping saddles. These theorems explain why "convergence" in nonconvex optimization often means reaching an approximate stationary point rather than proving global optimality.

5. **Saddles, plateaus, symmetries, and loss landscapes**

   Deep networks have many saddle points, flat directions, and equivalent parameterizations. Strict-saddle theory explains why negative-curvature saddles can be escaped by noise or perturbations, but deep-learning landscapes also contain non-isolated manifolds of equivalent solutions and long plateaus caused by scale symmetries, normalization, inactive units, or badly conditioned representations.

6. **Overparameterization and global convergence**

   Modern networks often have more parameters than training examples and can drive training loss nearly to zero. The main global-convergence theories include NTK or lazy-training limits, finite-width overparameterized convergence, mean-field limits, deep linear models, and matrix-factorization surrogates. These results are valuable, but each comes with a regime: lazy theory explains function-space kernel dynamics, while mean-field and feature-learning theories allow the representation itself to move.

7. **Implicit bias, interpolation, noise, and geometry**

   In overparameterized models, many parameters interpolate the data. Gradient methods therefore do more than minimize the loss; they select a particular interpolating solution. This section covers max-margin bias, norm and rank bias, post-interpolation dynamics, flat and sharp minima, stochastic gradient noise, batch size, neural collapse, mode connectivity, and spectral bias.

8. **Optimizers: momentum, adaptive methods, and Muon**

   Momentum changes the effective dynamics by accumulating velocity, adaptive methods such as AdaGrad, RMSProp, Adam, AMSGrad, and AdamW rescale coordinates, and matrix-structured methods such as K-FAC, Shampoo, SOAP, Muon, and Newton-Muon use richer geometry. These methods can speed convergence, stabilize training, or change implicit bias, but their theoretical guarantees are usually narrower than their empirical use.

9. **Multi-objective and multitask convergence**

   Many deep-learning objectives are sums of several losses: task losses, auxiliary losses, regularizers, alignment objectives, domain losses, and safety or preference terms. Optimizing a scalar sum is not the same as improving every component. The useful theory here is multi-objective optimization, Pareto stationarity, gradient conflict, and methods such as MGDA, PCGrad, CAGrad, and Nash-MTL.

10. **Learning-rate schedules and phase behavior**

   Learning-rate schedules are not cosmetic. Constant learning rates, step decay, cosine decay, inverse-square-root decay, one-cycle schedules, warmup, and Warmup-Stable-Decay (WSD) can lead to different convergence paths. Large learning rates produce catapult dynamics, progressive sharpening, and edge-of-stability behavior; decay phases can be smooth, or they can reveal instability through spikes and oscillations.

11. **LLM-specific convergence**

   Large language model training is usually judged by tokens, compute, wall-clock time, and scaling-law extrapolation rather than by asymptotic convergence to a training-set minimizer. Important phenomena include compute-optimal stopping, warmup and WSD schedules, data-mixture effects, grokking or delayed generalization, epoch-wise and model-wise double descent, and non-monotone or multi-phase loss curves. The phrase "double maximum of the loss curve" is not standard; a better survey label is "non-monotone and multi-phase loss curves."

12. **Methodology and open problems**

   A convergence claim should specify the optimizer, schedule, batch size, normalization, initialization, width/depth regime, data order, compute budget, and diagnostic. Open problems include feature-learning theory at scale, optimizer-dependent implicit bias, Muon-style orthogonalized updates, LLM-specific phase transitions, multitask convergence criteria that predict transfer, and diagnostics that connect training convergence to downstream generalization.

## 1. Notions of Convergence

For a deterministic optimizer on a fixed training objective, the cleanest notion is objective convergence:
$$
L_n(\theta_t)-\inf_\theta L_n(\theta)\to 0.
$$
This is the natural notion in convex optimization, where every local minimum is global and the objective gap has a clear meaning. In deep learning this definition is often too strong or too narrow. The infimum may be zero but not attained for cross-entropy on separable data, the same function may be represented by many parameter vectors, and training loss convergence may not imply validation or downstream convergence.

A weaker and more common nonconvex criterion is first-order stationarity:
$$
\|\nabla L_n(\theta_t)\|\to 0
\quad\text{or}\quad
\min_{0\leq t<T}\|\nabla L_n(\theta_t)\|^2 \leq \epsilon.
$$
For stochastic methods one usually proves a bound on $\mathbb{E}\|\nabla L_n(\theta_R)\|^2$ for a random iterate $R$. This is a useful baseline because it applies broadly to smooth nonconvex functions, but it says little about whether the stationary point is good. A saddle point, a bad local minimum, and a good interpolating solution can all have small gradient norm.

Second-order stationarity adds a Hessian condition:
$$
\|\nabla L_n(\theta)\|\leq \epsilon,
\qquad
\lambda_{\min}(\nabla^2 L_n(\theta))\geq -\sqrt{\epsilon}.
$$
This rules out points with strong negative curvature. It is the right target for strict-saddle theory, but it still does not imply global optimality unless the landscape has a benign structure. Deep networks also have flat directions from symmetries, so the Hessian may have many zero eigenvalues even near useful solutions.

Another distinction is parameter convergence versus function convergence. Because networks have permutation symmetries, scale symmetries, and redundant parameters, $\theta_t$ may not converge even when $f_{\theta_t}$ does. For example, in a ReLU network one can rescale adjacent layers by $c$ and $1/c$ without changing the represented function. For this reason, convergence in function space,
$$
\|f_{\theta_t}-f_\star\| \to 0,
$$
can be more meaningful than convergence in Euclidean parameter space. NTK theory makes this distinction explicit by describing training as kernel gradient descent on the vector of predictions rather than as convex optimization in weights.

A final distinction is training convergence versus generalization convergence. Training convergence asks whether $L_n(\theta_t)$ decreases on the sampled data. Generalization convergence asks whether validation loss, test loss, calibration, or downstream task quality improves. These can decouple after interpolation. The implicit-bias result of [Soudry, Hoffer, Nacson, Gunasekar, and Srebro 2018](https://jmlr.org/papers/v19/18-188.html) is a useful warning: for separable logistic regression, the loss keeps decreasing while the weight norm diverges, yet the normalized direction converges slowly toward the max-margin separator. Continuing to optimize after zero training error can therefore change the classifier even when the empirical classification error is already zero.

### What to Record in a Convergence Claim

When reading a convergence theorem or an empirical convergence plot, record:

- the objective being minimized: training loss, regularized loss, population risk, validation loss, or downstream metric;
- the convergence metric: objective gap, gradient norm, distance to optimum, sharpness, function-space error, or representation metric;
- the optimizer and schedule: GD, SGD, momentum, AdamW, Muon, cosine decay, WSD, warmup length, clipping, and weight decay;
- the regime: convex, nonconvex, overparameterized, NTK, mean-field, finite-width, large-batch, or LLM pretraining;
- the resource axis: steps, examples, tokens, FLOPs, or wall-clock time.

Without these details, statements such as "the model converges" or "Adam converges faster than SGD" are underspecified.

## 2. Empirical Convergence Phenomena

The simplest empirical plot is training loss versus optimization step. In small classical problems, one expects a monotone or nearly monotone decrease. Modern deep learning often violates this simple picture. Loss may spike during warmup, oscillate at large learning rate, plateau during a stable phase, or drop sharply only after a decay phase. Validation loss can peak and then recover, or it can improve long after training accuracy saturates. Therefore convergence experiments should usually plot several curves at once: training loss, validation loss, gradient norm, update norm $\|\theta_{t+1}-\theta_t\|$, learning rate, batch size, and some sharpness or Hessian-spectrum diagnostic.

A recurring empirical fact is that sufficiently large networks interpolate training data quickly. [Zhang, Bengio, Hardt, Recht, and Vinyals 2017](https://research.google/pubs/understanding-deep-learning-requires-rethinking-generalization/) showed that standard architectures can fit random labels, making it clear that low training loss alone cannot explain generalization. For convergence, the lesson is that successful optimization of the empirical objective is not the whole story. A training run may converge in the sense of zero training error but still be in a phase where representations, margins, calibration, or validation performance are changing.

Batch size is another major empirical variable. Small-batch SGD injects stochasticity, while large-batch training is closer to full-batch gradient descent and often needs learning-rate scaling, warmup, and schedule tuning. [Keskar et al. 2017](https://openreview.net/forum?id=H1oyRlYgg) associated very large batches with sharp-minima and generalization issues in some settings, while [McCandlish et al. 2018](https://arxiv.org/abs/1812.06162) introduced the gradient noise scale as a way to reason about critical batch size and training efficiency. The key convergence point is that increasing batch size can reduce gradient noise but does not indefinitely improve time-to-quality; beyond a critical scale, the computation becomes less efficient unless the schedule and optimizer are changed.

Wall-clock convergence is different from step convergence. An optimizer that reaches a target loss in fewer steps may still be slower if each step is expensive. This distinction matters for matrix preconditioners, second-order methods, and newer optimizers such as Muon. It also matters for distributed training: larger batches reduce synchronization frequency per token, but they may harm optimization if the learning-rate rule is not adjusted. A careful empirical comparison should report steps, examples or tokens, FLOPs, wall-clock time, and final quality.

Reproducibility is part of convergence. If two seeds reach different loss basins, or if one run diverges during warmup while another succeeds, the training procedure has not converged robustly in the practical sense. Seed sensitivity can come from initialization, data order, dropout, nondeterministic kernels, mixed precision, or optimizer state. For survey purposes, convergence evidence is stronger when it includes seed distributions rather than a single best run.

### Takeaway

Empirical convergence in deep learning is a multi-axis phenomenon. The training loss curve is necessary but not sufficient. A useful convergence experiment tracks optimization progress, stability, compute efficiency, and generalization together, and it states whether comparisons are made at equal steps, equal tokens, equal FLOPs, or equal wall-clock time.

## 3. Convex Optimization Background

Convex theory provides the baseline mathematical language. A differentiable function $L$ is convex if
$$
L(y)\geq L(x)+\nabla L(x)^\top(y-x)
$$
for all $x,y$. It is $\beta$-smooth if
$$
\|\nabla L(x)-\nabla L(y)\|\leq \beta\|x-y\|,
$$
or equivalently, when twice differentiable, $\nabla^2 L(x)\preceq \beta I$. Gradient descent with a fixed step $\eta\leq 1/\beta$ satisfies the descent inequality
$$
L(\theta_{t+1})\leq L(\theta_t)-\frac{\eta}{2}\|\nabla L(\theta_t)\|^2.
$$
For convex $L$, this yields the standard sublinear objective rate
$$
L(\bar\theta_T)-L(\theta^\star)\leq O\left(\frac{\|\theta_0-\theta^\star\|^2}{\eta T}\right),
$$
where $\bar\theta_T$ is an average iterate or, under additional assumptions, the last iterate. This is the simplest reference result: smooth convex optimization gets an $O(1/T)$ objective gap with full gradients.

If $L$ is also $\mu$-strongly convex, then gradient descent converges linearly. With a suitable constant step size,
$$
L(\theta_t)-L(\theta^\star)\leq (1-\eta\mu)^t\bigl(L(\theta_0)-L(\theta^\star)\bigr).
$$
This theorem explains why conditioning matters: the rate depends on the condition number $\kappa=\beta/\mu$. When $\kappa$ is large, some directions are steep and others are flat, so a learning rate stable for the steep direction makes slow progress along the flat direction.

Stochastic gradient descent replaces $\nabla L(\theta_t)$ by an unbiased estimator $g_t$ with variance. In convex settings, averaged SGD gives rates such as $O(1/\sqrt{T})$ for general convex objectives and $O(1/T)$ for strongly convex objectives under decaying step sizes and bounded variance assumptions. The survey by [Bottou, Curtis, and Nocedal 2018](https://doi.org/10.1137/16M1080173) is a standard reference for optimization methods in large-scale machine learning.

Acceleration changes the picture. Heavy-ball momentum, introduced by [Polyak 1964](https://doi.org/10.1016/0041-5553(64)90137-5), and Nesterov acceleration, introduced in Nesterov's 1983 convex-programming work, can improve convex rates for well-behaved objectives. In strongly convex smooth problems, accelerated methods improve the dependence on condition number from roughly $\kappa$ to $\sqrt{\kappa}$. This is a major classical reason to use momentum-like dynamics. However, the clean acceleration theory assumes convexity and carefully chosen parameters; in nonconvex deep networks, momentum can also amplify oscillations in sharp directions.

The Polyak-Lojasiewicz (PL) condition is especially relevant because it gives linear convergence without requiring convexity:
$$
\frac{1}{2}\|\nabla L(\theta)\|^2 \geq \mu\bigl(L(\theta)-L^\star\bigr).
$$
Under smoothness and the PL condition, gradient descent decreases the objective geometrically. Many overparameterized convergence proofs try to establish a local PL-like inequality or a uniform lower bound on a tangent kernel, which plays a similar role: small prediction error implies a sufficiently large gradient unless the model is already close to interpolation.

### What Convex Theory Contributes to Deep Learning

Convex theory is not a direct explanation of deep-network training, because neural-network objectives are nonconvex in the weights. Its contribution is conceptual. It defines the roles of smoothness, conditioning, strong convexity, stochastic variance, step-size stability, acceleration, and preconditioning. When a deep-learning phenomenon violates convex intuition, such as edge-of-stability training with nonmonotone loss, the convex baseline tells us exactly which assumption has failed.

## 4. General Nonconvex Optimization Background

For a smooth nonconvex objective, full-batch gradient descent still satisfies a descent inequality when $\eta\leq 1/\beta$:
$$
L(\theta_{t+1})\leq L(\theta_t)-\frac{\eta}{2}\|\nabla L(\theta_t)\|^2.
$$
Summing over $t=0,\ldots,T-1$ gives
$$
\min_{0\leq t<T}\|\nabla L(\theta_t)\|^2
\leq
\frac{2(L(\theta_0)-L_{\inf})}{\eta T}.
$$
This is the standard first-order nonconvex guarantee: gradient descent reaches a point with small gradient norm at rate $O(1/T)$ in squared gradient norm, assuming the objective is lower bounded and smooth.

Stochastic gradient methods have analogous bounds with slower rates. A representative result, developed in forms such as [Ghadimi and Lan 2013](https://doi.org/10.1137/120880811), says that for smooth nonconvex objectives and unbiased gradients with bounded variance, one can choose a random iterate $R$ so that
$$
\mathbb{E}\|\nabla L(\theta_R)\|^2 \leq O\left(\frac{1}{\sqrt{T}}\right)
$$
under appropriate step-size choices. This is useful because it applies broadly, but it is also weak: it guarantees approximate stationarity, not a good classifier, not a global minimum, and not generalization.

Second-order methods and perturbed first-order methods target approximate second-order stationarity. The strict-saddle literature assumes that every critical point is either a local minimum or has a direction of negative curvature. Under that assumption, small random perturbations or stochastic gradients can help escape saddles. [Jin, Ge, Netrapalli, Kakade, and Jordan 2017](https://arxiv.org/abs/1703.00887) show that perturbed gradient descent can find approximate local minima efficiently under smoothness and Hessian-Lipschitz assumptions. This line of work is important for deep learning because saddles are common in high-dimensional nonconvex problems.

A different route is to prove that all local minima are global or nearly global for a special model. Deep linear networks, matrix factorization, and certain overparameterized models are common test cases. [Kawaguchi 2016](https://proceedings.neurips.cc/paper/2016/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html) showed that deep linear networks have no bad local minima under suitable assumptions, though they do have saddle points. This does not solve nonlinear deep learning, but it gives a tractable setting where depth creates nonconvexity without necessarily creating bad local minima.

### What Nonconvex Theory Does and Does Not Say

Generic nonconvex theory explains why small gradient norm is a common convergence target and why saddle escape matters. It does not explain why large neural networks often reach near-zero training loss, why they generalize, or why large learning rates can remain stable beyond the classical monotone-descent regime. For those questions, one needs deep-learning-specific structure: overparameterization, architecture, initialization, normalization, stochasticity, and optimizer dynamics.

## 5. Saddles, Plateaus, Symmetries, and Loss Landscapes

Saddle points are unavoidable in high-dimensional nonconvex optimization. A point $\theta$ is a strict saddle if $\nabla L(\theta)=0$ and $\lambda_{\min}(\nabla^2 L(\theta))<0$. At such a point, there exists a local direction in which the objective decreases. Gradient descent can slow near a saddle because the gradient is small, but noise, stochastic gradients, or explicit perturbations can move the iterate into a negative-curvature direction.

Deep networks also have non-strict saddles and flat manifolds. Permuting hidden units often leaves the represented function unchanged. ReLU networks have positive-homogeneous scaling symmetries: multiplying incoming weights to a unit by $c>0$ and outgoing weights by $1/c$ preserves the function. Batch normalization and layer normalization add further scale invariances. These symmetries produce zero or near-zero Hessian eigenvalues that are not signs of poor optimization; they are artifacts of the parameterization.

Plateaus are different from isolated saddles. A plateau may have small gradients over a wide region, causing slow progress without a single critical point being responsible. Plateaus can arise from saturated activations, inactive ReLUs, badly scaled layers, poor signal propagation, or representation bottlenecks. Historically, vanishing gradients in very deep networks motivated residual connections and normalization layers. Residual networks, introduced by [He, Zhang, Ren, and Sun 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), can be viewed partly as an architectural response to convergence difficulty: skip connections create shorter gradient paths and make very deep models easier to optimize.

Loss-landscape geometry also includes sharpness and connectivity. Sharpness is often approximated by the largest Hessian eigenvalue
$$
\lambda_{\max}(\nabla^2 L(\theta)).
$$
In classical smooth optimization, stability of gradient descent near a quadratic minimum requires $\eta < 2/\lambda_{\max}$. In deep learning, however, sharpness is not invariant to reparameterization; [Dinh, Pascanu, Bengio, and Bengio 2017](https://proceedings.mlr.press/v70/dinh17b.html) showed that a network can be reparameterized to make a minimum appear arbitrarily sharp without changing the function. Therefore sharpness diagnostics are useful but must be interpreted with parameterization and normalization in mind.

Mode connectivity is another empirical challenge to simple landscape pictures. [Garipov et al. 2018](https://proceedings.neurips.cc/paper/2018/hash/be3087e74e9100d4bc4c6268cdbe8456-Abstract.html) and [Draxler et al. 2018](https://proceedings.mlr.press/v80/draxler18a.html) found low-loss paths connecting independently trained solutions. This suggests that many neural-network minima are not isolated basins separated by high barriers, but part of broad connected regions in functionally similar space. For convergence, this means that reaching a "minimum" may mean entering a low-loss region or manifold rather than locating a single point.

### Takeaway

Saddle escape is one important part of deep-learning convergence, but it is not the whole landscape story. Deep nets also have symmetry-induced flat directions, long plateaus, sharpness changes, connected low-loss regions, and architecture-dependent conditioning. A convergence survey should therefore treat saddles as one component of a broader geometry.

## 6. Overparameterization, Global Convergence, and Implicit Bias

The surprising empirical fact behind modern convergence theory is that large networks can often optimize nonconvex training objectives to near-zero loss. One explanation is overparameterization: when the model has enough degrees of freedom and is initialized in a favorable regime, the loss landscape near initialization can behave almost like a well-conditioned convex problem in prediction space.

The Neural Tangent Kernel (NTK) is the canonical lazy-training theory. [Jacot, Gabriel, and Hongler 2018](https://arxiv.org/abs/1806.07572) showed that in an infinite-width limit, the network's tangent kernel converges to a deterministic kernel and remains essentially constant during training. For squared loss on training predictions $u_t=f_{\theta_t}(X)$, the continuous-time dynamics become approximately
$$
\frac{d}{dt}(u_t-y)=-K(u_t-y),
$$
where $K$ is the NTK matrix on the training data. If $K$ is positive definite, the training error decays exponentially in function space. This turns a nonconvex parameter problem into a kernel regression convergence problem.

Finite-width overparameterized results prove related statements under explicit width and initialization conditions. For example, [Du, Lee, Li, Wang, and Zhai 2019](https://arxiv.org/abs/1811.03804) prove global convergence of gradient descent for overparameterized networks in regimes where the tangent kernel remains well conditioned. These results explain why sufficiently wide networks can converge from random initialization, but they are often closest to the lazy regime, where features do not move much.

Lazy training is not the only regime. [Chizat, Oyallon, and Bach 2019](https://arxiv.org/abs/1812.07956) clarified that lazy training is a scaling regime, not a universal description of all neural-network training. Mean-field theory studies limits in which the empirical distribution of parameters evolves, often through a Wasserstein gradient flow. [Mei, Montanari, and Nguyen 2018](https://arxiv.org/abs/1802.06015) and [Chizat and Bach 2018](https://arxiv.org/abs/1805.09545) are representative references for two-layer mean-field convergence and global optimization viewpoints. The main conceptual difference is that mean-field dynamics can represent feature learning, while pure NTK dynamics freeze the feature map at initialization.

Deep linear networks and matrix factorization give a tractable middle ground. They are nonconvex in parameters but linear in the represented map. They reveal phenomena such as balancedness across layers, rank bias, saddle structure, and implicit regularization. These models are not realistic image or language models, but they isolate effects that are hidden in fully nonlinear networks.

Overparameterization also creates an implicit-bias question. If many parameters achieve zero training loss, which one does gradient descent find? In linear separable classification, [Soudry et al. 2018](https://jmlr.org/papers/v19/18-188.html) proved convergence in direction to the hard-margin separator. In matrix factorization and deep linear models, gradient methods often prefer low-norm or low-rank structure. In neural networks, the implicit bias depends on architecture, parameterization, optimizer, learning rate, normalization, and data. Therefore convergence is not only about whether the loss reaches zero; it is also about which interpolating solution is selected.

### Post-Interpolation Dynamics

After training error reaches zero, cross-entropy can continue to decrease by increasing margins. This may improve robustness or generalization in some regimes and harm calibration or validation loss in others. Stochasticity also remains active after interpolation: SGD can diffuse along nearly flat manifolds, and the effective noise scale depends on batch size and learning rate. This helps explain why late training, weight decay, and learning-rate decay can still matter after the training labels are already fit.

Several late-training phenomena fit here. Neural collapse, studied by [Papyan, Han, and Donoho 2020](https://arxiv.org/abs/2008.08186), describes a terminal geometry in which class means and classifier weights approach a simplex-like configuration in the last-layer feature space. Spectral bias, studied by [Rahaman et al. 2019](https://proceedings.mlr.press/v97/rahaman19a.html), describes the tendency of neural networks to learn low-frequency components before high-frequency components in certain settings. These are convergence phenomena in representation space, not just in objective value.

### Takeaway

Overparameterization gives several routes to global training convergence, especially through NTK and related local-kernel arguments. But modern deep learning often relies on feature learning, optimizer-dependent implicit bias, and post-interpolation dynamics. A complete convergence story must say which regime it is analyzing.

## 7. Stochasticity, Batch Size, Normalization, and Parameterization

Mini-batch SGD can be written as
$$
g_t=\nabla L_n(\theta_t)+\xi_t,
$$
where $\xi_t$ is gradient noise from sampling a batch. The covariance of $\xi_t$ depends on the batch size, the data distribution, and the current parameters. Small batches create noisier updates, while large batches make the dynamics closer to full-batch gradient descent. In continuous-time approximations, SGD is often compared to a stochastic differential equation, but the analogy depends on the scaling of learning rate, batch size, and gradient covariance.

The noise view explains why batch-size scaling is not unlimited. If the batch is much smaller than the gradient-noise scale, increasing batch size can reduce noise and improve hardware efficiency without changing the optimization path too much. If the batch is already beyond the critical scale, further increases give less benefit per example and may require learning-rate or schedule changes. [McCandlish et al. 2018](https://arxiv.org/abs/1812.06162) formalized this idea empirically through the gradient noise scale.

Normalization changes convergence by changing geometry. Batch normalization, introduced by [Ioffe and Szegedy 2015](https://proceedings.mlr.press/v37/ioffe15.html), layer normalization, introduced by [Ba, Kiros, and Hinton 2016](https://arxiv.org/abs/1607.06450), and weight normalization, introduced by [Salimans and Kingma 2016](https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html), all change how gradients scale through a model. They often allow larger learning rates and reduce sensitivity to initialization, but they also introduce scale invariances that complicate standard Euclidean convergence analysis.

Parameterization matters even when the represented function class is unchanged. The same function-space model can have different gradient dynamics under different scalings of weights, widths, or residual branches. The maximal-update parameterization, or $\mu$P, developed by [Yang and Hu 2021](https://arxiv.org/abs/2011.14522), is one influential attempt to choose width scalings so that hyperparameters transfer more predictably across model sizes. For convergence, the lesson is that "the network" is not fully specified by the architecture; the parameterization determines the size and direction of gradient updates as width changes.

Gradient clipping is another practical convergence tool, especially in recurrent networks and transformers. It replaces an update $g_t$ by a clipped version when $\|g_t\|$ is too large. This is not just a numerical trick: clipping changes the optimizer's direction and can interact with Adam moments, mixed precision, and large-batch training. Loss spikes in large-scale training are often handled by clipping, learning-rate adjustment, skipped updates, or checkpoint rollback, which are practical convergence interventions rather than purely theoretical ones.

### Takeaway

Stochasticity, normalization, and parameterization decide the effective geometry of training. They determine which learning rates are stable, how noise scales with batch size, and whether a theoretical width limit resembles the actual training run.

## 8. Optimizers: Momentum, Adaptive Methods, and Muon

Momentum introduces a velocity variable. A simple heavy-ball update is
$$
v_{t+1}=\beta v_t+\nabla L(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta v_{t+1}.
$$
On well-conditioned convex objectives, momentum can accelerate convergence by carrying progress along shallow directions. On ill-conditioned objectives, it can also create oscillations in steep directions. In deep learning this tradeoff is visible near sharp valleys: momentum can help cross flat plateaus, but it can also make the effective step too aggressive when the local curvature changes quickly.

Adaptive gradient methods rescale coordinates. AdaGrad accumulates squared gradients and gives larger effective steps to coordinates with small historical gradients. RMSProp uses an exponential moving average of squared gradients. Adam, introduced by [Kingma and Ba 2015](https://arxiv.org/abs/1412.6980), combines momentum and RMS-style adaptivity:
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\qquad
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$
Adam is extremely effective in transformers and many nonstationary training settings, but its convergence theory is delicate. [Reddi, Kale, and Kumar 2018](https://openreview.net/forum?id=ryQu7f-RZ) showed that the original Adam update can fail to converge in some convex settings and proposed AMSGrad. AdamW, introduced by [Loshchilov and Hutter 2019](https://openreview.net/forum?id=Bkg6RiCqY7), decouples weight decay from the adaptive gradient step and became the standard optimizer for many large-scale language and vision models.

Preconditioned and second-order methods use richer geometry. K-FAC approximates natural-gradient curvature with Kronecker factors; see [Martens and Grosse 2015](https://proceedings.mlr.press/v37/martens15.html). Shampoo, introduced by [Gupta, Koren, and Singer 2018](https://proceedings.mlr.press/v80/gupta18a.html), uses matrix preconditioners for tensor parameters. These methods can reduce ill-conditioning, but they add computational and implementation complexity. Their convergence should be evaluated in terms of time-to-quality, not only steps-to-quality.

### Muon and Orthogonalized Updates

Muon is a newer optimizer for matrix-shaped hidden-layer parameters. The primary description is [Jordan et al. 2024](https://kellerjordan.github.io/posts/muon/). Muon first forms a momentum update for a weight matrix, then approximately orthogonalizes that update using a Newton-Schulz iteration. In idealized notation, if the momentum update matrix has singular value decomposition
$$
M=U\Sigma V^\top,
$$
Muon replaces it by
$$
\operatorname{msgn}(M)=UV^\top,
$$
the nearest semi-orthogonal matrix in Frobenius norm under the relevant rectangular constraint. The actual implementation uses a small fixed number of Newton-Schulz steps instead of an exact SVD, because the goal is a fast GPU-friendly approximation.

Muon is usually used as a hybrid optimizer: apply Muon to internal matrix weights and use AdamW for embeddings, output heads, biases, gains, layer-norm parameters, and other scalar or vector parameters. This matters for convergence comparisons. A "Muon run" is often not pure Muon on every parameter; it is a structured combination of orthogonalized matrix updates and AdamW-style updates.

The theoretical interpretation of Muon is still developing. One useful connection is to accumulation-free Shampoo: orthogonalizing $G$ resembles a spectral-norm steepest-descent or matrix-preconditioned step that equalizes singular directions. The 2026 [Newton-Muon paper by Du and Su](https://arxiv.org/abs/2604.01472) gives a more explicit surrogate-model derivation. It argues that standard Muon can be interpreted as neglecting a right preconditioning factor induced by the input second moment, and proposes Newton-Muon with an update of the form
$$
W \leftarrow W-\eta\,\operatorname{msgn}\bigl(G(ZZ^\top)^{-1}\bigr)
$$
up to momentum and weight decay, where $Z$ stacks layer inputs.

Muon should be treated carefully in a survey. Empirically it has strong speed-to-loss results in some NanoGPT-style and large-batch settings, but the evidence base is still newer and more heterogeneous than for SGD, momentum, or AdamW. The right questions are: which parameters use Muon, what auxiliary optimizer is used, how much extra wall-clock cost does orthogonalization add, how does it scale with matrix shape, how sensitive is it to batch size, and what implicit bias does the semi-orthogonal update impose?

### Takeaway

Optimizers are convergence mechanisms, not interchangeable implementation details. Momentum changes temporal dynamics, Adam changes coordinate geometry, AdamW changes the role of weight decay, matrix preconditioners change curvature handling, and Muon changes the singular-value geometry of matrix updates. Comparisons should report both step efficiency and wall-clock efficiency.

## 9. Multi-Objective and Multitask Convergence

Many deep-learning objectives are written as a scalar sum
$$
L(\theta)=\sum_{k=1}^K \lambda_k L_k(\theta),
$$
where $L_k$ might be a task loss, auxiliary loss, regularizer, alignment loss, domain loss, or safety objective. If the weights $\lambda_k$ are fixed, the optimization problem is a standard scalar problem. But convergence of $L$ does not imply convergence of every component $L_k$. One task can improve while another gets worse, and changing the scale of one loss can dominate the gradient even if it is not more important.

The multi-objective viewpoint keeps the vector loss
$$
\mathbf{L}(\theta)=(L_1(\theta),\ldots,L_K(\theta)).
$$
A point is Pareto stationary if there is no descent direction that decreases all objectives to first order. A useful first-order condition is
$$
\min_{\alpha\in\Delta_K}\left\|\sum_{k=1}^K \alpha_k\nabla L_k(\theta)\right\|^2=0,
$$
where $\Delta_K$ is the probability simplex. If the minimum is nonzero, then some convex combination of gradients gives a common descent direction. This is the basis for the multiple-gradient descent algorithm (MGDA), studied for multitask learning by [Sener and Koltun 2018](https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html).

Gradient conflict is central. Two objectives conflict when their gradients have negative cosine similarity:
$$
\nabla L_i(\theta)^\top \nabla L_j(\theta)<0.
$$
In that case, a step that decreases one loss may increase the other. PCGrad, introduced by [Yu et al. 2020](https://arxiv.org/abs/2001.06782), projects gradients to reduce conflicting components. CAGrad, introduced by [Liu et al. 2021](https://arxiv.org/abs/2110.14048), balances average descent with conflict aversion. Nash-MTL, introduced by [Navon et al. 2022](https://proceedings.mlr.press/v162/navon22a.html), frames multitask training as a bargaining problem.

Regularization terms also create multi-objective behavior. Weight decay, auxiliary classifiers, contrastive losses, preference losses, KL penalties, and safety constraints all pull the model in different directions. In LLM training, supervised fine-tuning, reinforcement learning from preferences, and safety training often combine objectives whose gradients are not aligned. A scalar aggregate loss can look stable while one component degrades. Therefore convergence diagnostics should track each component separately and, when possible, track gradient cosines or Pareto-stationarity residuals.

### Takeaway

For multitask and multi-loss training, "the loss converged" is not enough. The survey should ask which weighted scalar objective converged, whether individual objectives improved, whether gradients conflict, and whether the final point is Pareto reasonable for the intended tradeoff.

## 10. Learning-Rate Schedules and Phase Behavior

The learning rate controls both speed and stability. For a quadratic objective $L(\theta)=\frac{1}{2}\theta^\top H\theta$ with largest eigenvalue $\lambda_{\max}$, gradient descent is stable only when
$$
0<\eta<\frac{2}{\lambda_{\max}}.
$$
This classical condition motivates many intuitions about sharpness and step size. But deep networks are not fixed quadratics: the Hessian changes along the path, the network function changes nonlinearly, stochastic gradients add noise, and normalization changes scale. As a result, large learning rates can produce useful phases that are not captured by monotone-descent theory.

Common schedules include constant learning rate, step decay, exponential decay, cosine decay, inverse-square-root decay, polynomial decay, cyclical learning rates, one-cycle schedules, and warm restarts. Cosine annealing and warm restarts were popularized by [Loshchilov and Hutter 2017](https://openreview.net/forum?id=Skq89Scxx). The one-cycle and super-convergence viewpoint was developed by [Smith and Topin 2019](https://arxiv.org/abs/1708.07120). In transformers, inverse-square-root schedules and warmup became standard after [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), and AdamW with warmup plus cosine or WSD is now common in large-scale training.

Warmup is the initial phase in which $\eta_t$ increases from a small value to a target value. It is especially important for Adam-style optimizers, large batches, transformers, and mixed-precision training. Warmup prevents early optimizer states and activations from producing unstable updates before gradients and moments have reached a reasonable scale. Too little warmup can cause divergence or loss spikes; too much warmup wastes steps at low learning rate.

### Catapult and Edge of Stability

The catapult mechanism, studied by [Lewkowycz, Bahri, Dyer, Sohl-Dickstein, and Gur-Ari 2020](https://arxiv.org/abs/2003.02218), describes a large-learning-rate phase in which the loss can initially increase sharply and then decrease to a useful solution. This is already outside the classical small-step monotone-descent picture. The catapult phase helps explain why optimal performance can occur at learning rates that would look too large from a fixed-quadratic analysis.

Edge of stability is a related but distinct phenomenon. [Cohen, Kaur, Li, Kolter, and Talwalkar 2021](https://arxiv.org/abs/2103.00065) observed that full-batch gradient descent on neural networks often enters a regime where the sharpness hovers near or just above $2/\eta$, the short-term loss is nonmonotone, but the long-term trend still decreases. This is surprising because the classical quadratic condition predicts instability near that boundary. Follow-up work such as [Arora, Li, and Panigrahi 2022](https://arxiv.org/abs/2205.09745) and [Agarwala, Pedregosa, and Pennington 2022](https://arxiv.org/abs/2210.04860) analyzes mechanisms for edge-of-stability and progressive-sharpening behavior.

Progressive sharpening means that the largest Hessian eigenvalue tends to increase during training until it reaches the edge set by the learning rate. This is not merely a property of the final minimum; it is a path-dependent phenomenon. The optimizer and learning rate shape the curvature of the region that training visits.

### Smooth Decay, Unstable Decay, and WSD

A decay phase lowers the learning rate after a period of high-rate training. In a smooth decay phase, loss and update norms decrease steadily as the iterate settles into a lower-noise or lower-oscillation region. In an unstable decay phase, the loss may spike, oscillate, or improve only after transient instability. Causes include too-large peak learning rate, poorly tuned Adam moments, delayed curvature increase, gradient clipping interactions, data distribution shifts, and mixed-precision numerical effects.

Warmup-Stable-Decay (WSD) makes the phase structure explicit:
$$
\eta(t)=
\begin{cases}
\eta_{\max}\frac{t}{T_w}, & 0\leq t<T_w,\\
\eta_{\max}, & T_w\leq t<T_s,\\
\eta_{\max} f\left(\frac{t-T_s}{T_d}\right), & T_s\leq t\leq T_s+T_d,
\end{cases}
$$
where $f$ is a decreasing cooldown shape. WSD was introduced into small language model training in the MiniCPM work of [Hu et al. 2024](https://arxiv.org/abs/2404.06395). [Wen, Li, Wang, Hall, Liang, and Ma 2025](https://openreview.net/forum?id=m51BgoqvbP) give a "river valley" interpretation: during the stable phase, a high learning rate keeps the iterate oscillating across steep directions while still making progress along flatter valley directions; during decay, the oscillation shrinks and the loss drops. [Dremov, Hagele, Kosson, and Jaggi 2025](https://openreview.net/forum?id=ZnSYEcZod3) focus specifically on cooldown dynamics and show that cooldown shape and AdamW hyperparameters can materially affect final transformer performance.

This explains why WSD can produce a nontraditional loss curve: the stable-phase loss may remain elevated, then the decay phase reveals progress by reducing oscillations. Therefore a flat or high stable-phase loss does not necessarily mean no learning is occurring. Conversely, a sharp decay-phase drop does not mean the model learned everything only at the end; it may mean that earlier high-rate training moved along useful directions while hiding progress behind oscillation.

### Takeaway

Learning-rate schedules are part of the optimizer, not post-processing. Large learning rates can create catapult and edge-of-stability dynamics; decay can either smooth convergence or expose instability; WSD separates progress along high-rate and low-rate directions. Any convergence claim for deep learning should specify the schedule.

## 11. LLM-Specific Convergence

LLM training changes the meaning of convergence because the run is usually stopped for compute reasons, not because the training objective has reached a mathematical optimum. The resource axis is tokens and FLOPs. The relevant question is often: for a fixed compute budget, which model size, dataset size, optimizer, schedule, and stopping point gives the lowest validation loss or best downstream performance?

Scaling laws make this explicit. [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) modeled language-model loss as a power law in model size, data size, and compute, and emphasized that compute-efficient training can stop before full convergence on the training data. [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) revised the compute-optimal tradeoff in the Chinchilla work, arguing for training smaller models on more tokens than earlier scaling implied. From a convergence perspective, these papers shift attention from asymptotic minimization to finite-compute trajectories.

LLM convergence is also data-mixture convergence. A validation loss averaged over a mixture can improve while a minority domain worsens. Curriculum, deduplication, code/math mixture, multilingual mixture, long-context data, and instruction data can all change the gradient direction. Therefore language-model convergence diagnostics should include per-domain loss, downstream slices, memorization measures, and data-order effects when possible.

### Grokking and Delayed Generalization

Grokking refers to delayed generalization: the model first memorizes or fits the training set, then much later generalizes. The term comes from [Power, Burda, Edwards, Babuschkin, and Misra 2022](https://arxiv.org/abs/2201.02177), who studied small algorithmic datasets where test performance improves long after training performance is high. Grokking is a convergence phenomenon because the train loss or train accuracy can appear saturated while the model's solution is still changing.

Recent work has looked for analogous behavior in larger models. [Li, Fan, and Zhou 2026](https://arxiv.org/abs/2506.21551) study grokking in practical LLM pretraining and frame it as a memorization-to-generalization transition that can be monitored without ordinary test evaluation in their MoE setting. This does not mean every LLM run groks in the original toy sense. It means delayed generalization is now a serious convergence topic for large-scale pretraining, especially when different data groups or expert pathways develop asynchronously.

### Non-Monotone and Multi-Phase Loss Curves

The phrase "double maximum of the loss curve" is not a standard term. It is better to group such observations under non-monotone or multi-phase loss curves. Several mechanisms can produce local peaks or two-stage shapes:

- catapult dynamics can create an early loss spike followed by rapid decrease;
- edge-of-stability training can create short-timescale oscillations around a decreasing trend;
- WSD can keep loss elevated during the stable phase and then drop sharply during cooldown;
- grokking can show delayed validation improvement after training metrics appear saturated;
- double descent can make test error worsen and then improve as model size, dataset size, or training time changes.

Deep double descent, studied by [Nakkiran et al. 2020](https://openreview.net/forum?id=B1g5sA4twr), is especially important because it includes epoch-wise double descent: training longer can first hurt and then help test performance in some regimes. This is not simply a loss curve with two maxima; it is a relationship between effective model complexity and generalization error. In a convergence survey, double descent should be separated from training-loss spikes and WSD cooldown drops, even though all three can look non-monotone in plots.

### Practical LLM Convergence Diagnostics

For LLMs, report at least training loss, validation loss, per-domain validation loss, learning rate, gradient norm, update norm, loss spikes, tokens processed, FLOPs, wall-clock time, and checkpoint quality. For multitask or instruction-tuned models, report component losses and representative downstream metrics. For MoE models, also track routing load, expert utilization, and routing stability. For long-context or curriculum runs, report loss by context length or data stage. These diagnostics are not optional details; they determine what kind of convergence claim is being made.

### Takeaway

LLM convergence is finite-compute, schedule-dependent, and data-mixture-dependent. Scaling laws, WSD, grokking, double descent, loss spikes, and checkpoint branching are all part of the convergence picture. The right question is usually not "did training converge?" but "which training trajectory gives the best model under the available compute and data constraints?"

## 12. Methodology and Open Problems

A clean convergence experiment should control one axis at a time. If the question is optimizer convergence, keep architecture, batch size, schedule, weight decay, gradient clipping, data order, and compute fixed, or explain why they are retuned. If the question is schedule convergence, compare schedules at equal token budgets and also at equal wall-clock budgets when schedules change throughput. If the question is large-batch scaling, report the critical batch size, learning-rate scaling rule, warmup, and whether final quality or time-to-target is the metric.

For theory, the most important methodological step is to state the regime. Convex, PL, strict-saddle, NTK, mean-field, finite-width, deep linear, matrix factorization, and transformer pretraining are different mathematical objects. A theorem in one regime can still be useful, but it should not be presented as a theorem about all of deep learning. The same caution applies to empirical evidence: a NanoGPT speedrun, CIFAR-10 classifier, ResNet on ImageNet, and 7B-parameter MoE pretraining run test different convergence phenomena.

Several open problems are central:

- feature-learning convergence beyond lazy NTK theory;
- optimizer-dependent implicit bias for AdamW, Shampoo-style methods, Muon, and hybrids;
- theory for large learning rates, edge-of-stability, and schedule-dependent phases in realistic architectures;
- convergence diagnostics that predict downstream generalization rather than only training loss;
- multi-objective convergence criteria that predict transfer and avoid task collapse;
- LLM-specific delayed generalization, data-mixture dynamics, and checkpoint-branching theory;
- finite-width and finite-compute theory that matches practical stopping rules.

### Final Takeaway

Convergence of deep learning is not one theorem. The classical backbone is convex and nonconvex optimization, but modern practice adds overparameterization, implicit bias, stochasticity, normalization, adaptive and matrix-structured optimizers, large-learning-rate phase behavior, multitask objectives, and LLM-scale compute constraints. A useful survey keeps these regimes separate, states the convergence metric precisely, and treats the training trajectory itself as an object of study.
