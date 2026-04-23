# Scaling limit of deep networks

## Outline

A scaling limit is an asymptotic description of a neural network family as one or more structural quantities become large. The quantity may be width, depth, input dimension, sample size, number of heads, key/query dimension, or context length. Different choices lead to different mathematical objects: Gaussian processes, deterministic kernels, nonlinear particle systems, stochastic corrections, dynamical mean-field equations, or continuum attention dynamics. The main lesson of this section is that there is no single "the" scaling limit of neural networks. One must specify what is being scaled, how parameters and learning rates are normalized, whether training is included, and which architecture is under study.

1. **Initialization-only infinite-width prior / NNGP limit**

   Before training, wide randomly initialized networks often converge to Gaussian processes. This is the neural-network Gaussian-process (NNGP) or infinite-width prior limit. It describes the random function induced by the architecture and initialization, not the effect of gradient descent. It begins with [Neal 1996](https://doi.org/10.1007/978-1-4612-0745-0) and [Williams 1997](https://doi.org/10.1162/neco.1997.9.6.1299), and modern deep versions include [Lee et al. 2018](https://arxiv.org/abs/1711.00165), [Matthews et al. 2018](https://arxiv.org/abs/1804.11271), [Novak et al. 2019](https://arxiv.org/abs/1810.05148), and Tensor Programs such as [Yang 2019](https://arxiv.org/abs/1910.12478).

2. **Neural tangent kernel / lazy / linearized training limit**

   The neural tangent kernel (NTK) is the training-dynamics counterpart of the infinite-width prior limit. With the usual NTK scaling and fixed depth, the empirical tangent kernel becomes deterministic and remains nearly constant during training, so gradient descent on parameters becomes kernel gradient descent in function space. Core references are [Jacot, Gabriel, and Hongler 2018](https://arxiv.org/abs/1806.07572), [Lee et al. 2019](https://arxiv.org/abs/1902.06720), [Chizat, Oyallon, and Bach 2019](https://arxiv.org/abs/1812.07956), and [Woodworth et al. 2020](https://arxiv.org/abs/2002.09277).

3. **Mean-field / particle / PDE limits**

   Mean-field limits are not the same as NTK limits. A two-layer network can be written as a finite empirical measure over neurons, and as width grows that empirical measure can converge to a distribution evolving by a nonlinear Wasserstein gradient flow or related PDE. This produces feature-learning dynamics rather than a frozen kernel. Foundational papers include [Mei, Montanari, and Nguyen 2018](https://arxiv.org/abs/1804.06561), [Sirignano and Spiliopoulos 2018](https://arxiv.org/abs/1805.01053), [Rotskoff and Vanden-Eijnden 2018](https://arxiv.org/abs/1805.00915), and [Chizat and Bach 2018](https://arxiv.org/abs/1805.09545). Multilayer extensions form a separate branch, including [Sirignano and Spiliopoulos 2019](https://arxiv.org/abs/1903.04440), [Nguyen 2019](https://arxiv.org/abs/1902.02880), and [Nguyen and Pham 2020](https://arxiv.org/abs/2001.11443).

4. **Feature-learning infinite-width limits**

   Standard NTK limits often freeze hidden representations too strongly. A separate literature studies infinite-width limits in which features still move at leading order. This includes mean-field scalings, maximal-update parameterizations, and dynamical mean-field theories. Representative references are [Yang and Hu 2020](https://arxiv.org/abs/2011.14522), [Yang et al. 2022](https://arxiv.org/abs/2203.03466), and [Bordelon and Pehlevan 2022](https://arxiv.org/abs/2205.09653).

5. **Parameterization, universality, and architecture-general scaling theory**

   The limiting object is controlled by parameterization. Standard, NTK, mean-field, and muP scalings can describe different asymptotic networks even when the finite network architecture looks the same. Tensor Programs provide a unified framework for GP, NTK, and feature-learning limits across MLPs, CNNs, RNNs, attention, normalization layers, and transformers; see [Tensor Programs I](https://arxiv.org/abs/1910.12478), [Tensor Programs II](https://arxiv.org/abs/2006.14548), and [Tensor Programs V / muTransfer](https://arxiv.org/abs/2203.03466).

6. **Finite-width corrections and fluctuation theory**

   Infinite-width limits are leading-order descriptions. Finite networks have fluctuations around those limits, often of order $m^{-1/2}$ or $m^{-1}$ depending on the observable. This branch studies random empirical NTKs, central-limit corrections, diagrammatic expansions, and how depth changes finite-width error. Useful references include [Hanin and Nica 2019](https://arxiv.org/abs/1909.05989), [Dyer and Gur-Ari 2019](https://arxiv.org/abs/1909.11304), [Pham and Nguyen 2021](https://proceedings.neurips.cc/paper/2021/hash/2639ba2137371773aa1e64e7735cdb30-Abstract.html), and [Bordelon and Pehlevan 2023](https://arxiv.org/abs/2304.03408).

7. **Depth limits, signal propagation, and edge-of-chaos theory**

   Width is not the only asymptotic variable. When depth grows, forward correlations, backward gradients, and Jacobian spectra can converge, explode, vanish, or enter critical regimes. This line includes edge-of-chaos theory, dynamical isometry, ResNet scaling, and continuous-depth limits. Representative references are [Schoenholz et al. 2016](https://arxiv.org/abs/1611.01232), [Yang and Schoenholz 2017](https://arxiv.org/abs/1712.08969), [Xiao et al. 2018](https://arxiv.org/abs/1806.05393), and [Chen et al. 2018](https://arxiv.org/abs/1806.07366).

8. **Joint width-depth and high-dimensional proportional limits**

   Many classical theorems take width to infinity at fixed depth and fixed data dimension. Modern networks are not naturally in that regime. Joint limits send depth and width to infinity together, or send width, input dimension, and sample size to infinity at comparable rates. These limits can produce non-Gaussian laws, random-matrix formulas, and feature-learning effects absent from fixed-depth GP/NTK theory. See [Li, Nica, and Roy 2021](https://arxiv.org/abs/2106.04013), [Mei and Montanari 2019](https://arxiv.org/abs/1908.05355), and [Ba et al. 2022](https://arxiv.org/abs/2205.01445).

9. **Transformer width, head, and depth asymptotics**

   For transformers, "infinite width" is too coarse. One can send model dimension, key/query dimension, value dimension, MLP width, number of heads, or depth to infinity, and the limits are not interchangeable. Attention also introduces non-Gaussian structure because softmax couples tokens through random similarity scores. Useful references include [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), [Bordelon, Chaudhry, and Pehlevan 2024](https://arxiv.org/abs/2405.15712), [Noci et al. 2022](https://arxiv.org/abs/2206.03126), and [Sakai, Karakida, and Imaizumi 2025](https://arxiv.org/abs/2506.00846).

10. **Transformer mean-field dynamics**

    Attention can be viewed as an interacting-particle system over token representations. In this line, the empirical distribution of tokens or hidden states evolves under nonlinear interaction dynamics. The resulting limit can explain clustering, consensus, rank collapse, metastability, and phase transitions. Recent references include [Kim and Suzuki 2024](https://arxiv.org/abs/2402.01258), [Poc-Lopez and Aguilera 2024](https://arxiv.org/abs/2406.07247), [Burger et al. 2025](https://arxiv.org/abs/2501.03096), and [Rigollet 2025](https://arxiv.org/abs/2512.01868).

11. **Transformer long-context / sequence-length limits**

    Long-context limits send the sequence length $T$ to infinity. This is different from width and different from ordinary length extrapolation experiments, but the topics interact strongly. The relevant questions include how positional encodings extrapolate, how attention logits should be rescaled with $T$, whether attention becomes uniform or too sharp, and whether token representations collapse or remain separated. Core references include ALiBi from [Press, Smith, and Lewis 2021](https://arxiv.org/abs/2108.12409), RoPE from [Su et al. 2021](https://arxiv.org/abs/2104.09864), length-generalization studies such as [Kazemnejad et al. 2023](https://arxiv.org/abs/2305.19466) and [Huang et al. 2024](https://arxiv.org/abs/2410.02140), RoPE-bound work such as [Men et al. 2024](https://arxiv.org/abs/2405.14591), and critical attention-scaling theory such as [Chen, Lin, Polyanskiy, and Rigollet 2026](https://openreview.net/forum?id=7SLtElfqCW).

12. **Non-Gaussian priors and heavy-tailed variants**

    The usual NNGP story relies on finite-variance central-limit behavior. With heavy-tailed weights or different normalization, the infinite-width prior can converge to a stable non-Gaussian process rather than a Gaussian process. This is a smaller but real branch of the literature; see [Favaro, Fortini, and Peluchetti 2020](https://arxiv.org/abs/2003.00394) and [Jung et al. 2021](https://arxiv.org/abs/2106.11064).

## 1. What a Scaling Limit Is

Consider a family of networks $f_{\theta}^{(s)}$, where $s$ denotes one or more scale parameters: width $m$, depth $L$, input dimension $d$, training-set size $n$, number of heads $H$, key/query dimension $d_k$, or context length $T$. A scaling-limit theorem chooses a normalization of weights, biases, residual branches, learning rates, and sometimes time, then asks whether
$$
f_{\theta}^{(s)} \;\Longrightarrow\; F_\infty
$$
in distribution, in probability, or along a training trajectory as $s\to\infty$. The limit $F_\infty$ may be a random function, a deterministic kernel model, a PDE, a stochastic process, or a set of order-parameter equations.

The phrase "scaling limit" is therefore incomplete without four choices. First, which variables go to infinity? Second, how are the parameters initialized and normalized? Third, is the network observed only at initialization, or is training included? Fourth, is the architecture an MLP, CNN, ResNet, RNN, or transformer? Changing any of these can change the answer. For example, an MLP at initialization with finite-variance weights may converge to a Gaussian process, while the same MLP under gradient flow in NTK scaling may converge to kernel regression, and a differently parameterized two-layer model may converge to nonlinear mean-field dynamics.

This distinction is especially important because the word "large" hides multiple asymptotics. An infinite-width limit at fixed data dimension is not a high-dimensional statistical limit. A long-context transformer limit is not a wide-transformer limit. A fixed-depth GP theorem does not automatically describe a model whose depth grows with width. The purpose of the taxonomy above is to keep these regimes separate.

## 2. Infinite-Width Priors and NNGP Limits

The oldest scaling limit of neural networks is the infinite-width Bayesian prior. For a one-hidden-layer network
$$
f_m(x)=\frac{1}{\sqrt{m}}\sum_{i=1}^{m} a_i \sigma(w_i\cdot x+b_i),
$$
with independent finite-variance parameters, the central limit theorem suggests that for any finite set of inputs $x_1,\ldots,x_N$, the vector $(f_m(x_1),\ldots,f_m(x_N))$ converges in distribution to a centered Gaussian vector. The covariance is determined by the activation, the input inner products, and the initialization law:
$$
K(x,x')=\mathbb{E}_{w,b,a}\bigl[a^2\sigma(w\cdot x+b)\sigma(w\cdot x'+b)\bigr].
$$
This is a Gaussian process limit because every finite-dimensional marginal converges to a Gaussian vector with a consistent covariance kernel.

[Neal 1996](https://doi.org/10.1007/978-1-4612-0745-0) made this Bayesian neural-network connection explicit for wide networks, building on the idea that a random neural network can define a distribution over functions. [Williams 1997](https://doi.org/10.1162/neco.1997.9.6.1299) studied computation with these infinite networks. Modern deep versions showed that the same idea extends layer by layer: [Lee et al. 2018](https://arxiv.org/abs/1711.00165) and [Matthews et al. 2018](https://arxiv.org/abs/1804.11271) analyze deep fully connected networks, while [Novak et al. 2019](https://arxiv.org/abs/1810.05148) and related work derive convolutional GP limits.

For a fully connected network, the covariance recursion illustrates the idea. Let $K^\ell(x,x')$ be the covariance between preactivations at layer $\ell$ for inputs $x$ and $x'$. With suitable independent Gaussian weights and biases,
$$
K^{\ell+1}(x,x')=\sigma_b^2+\sigma_w^2\mathbb{E}_{(u,v)\sim \mathcal{N}(0,\Sigma^\ell)}[\phi(u)\phi(v)],
$$
where
$$
\Sigma^\ell =
\begin{pmatrix}
K^\ell(x,x) & K^\ell(x,x')\\
K^\ell(x',x) & K^\ell(x',x')
\end{pmatrix}.
$$
Thus the architecture induces a deterministic kernel recursion even though the finite network is random. At initialization, the wide network behaves like a sample from a GP with kernel $K^L$.

The NNGP limit is useful because it separates architecture from training. It says what random functions a wide architecture places prior mass on. If Bayesian inference is then performed in the infinite-width limit, prediction reduces to Gaussian-process regression or classification with the corresponding architecture-induced kernel. This gives a clean theoretical object, but it also removes feature learning: the kernel is fixed by the random initialization and does not adapt through gradient descent unless one studies training separately.

NNGP theory is not limited to MLPs. [Yang 2019](https://arxiv.org/abs/1910.12478) introduced Tensor Programs as a general method for proving GP limits across broad computation graphs, including recurrent networks, normalization layers, pooling, and attention-like operations under appropriate assumptions. This architecture-general perspective matters because a hand-written covariance recursion is manageable for MLPs and CNNs but quickly becomes cumbersome for modern networks.

Finally, the Gaussian conclusion depends on finite-variance central-limit behavior. If weights are heavy-tailed and normalized differently, the limiting process can be stable and non-Gaussian. [Favaro, Fortini, and Peluchetti 2020](https://arxiv.org/abs/2003.00394) and [Jung et al. 2021](https://arxiv.org/abs/2106.11064) show that heavy-tailed infinitely wide networks can converge to alpha-stable processes. This branch is narrower than the GP literature, but it is a useful warning: "infinite width" does not automatically mean "Gaussian"; it means "whatever limit the normalization and initialization imply."

### Takeaway

The NNGP limit is an initialization or Bayesian-prior limit. It converts a wide random network into a kernel-defined Gaussian process, but by itself it does not explain representation learning during gradient descent.

## 3. Neural Tangent Kernel and Lazy Training

The neural tangent kernel describes how the network output changes under parameter updates. For a network $f_\theta$, define
$$
\Theta_\theta(x,x')=\nabla_\theta f_\theta(x)\cdot \nabla_\theta f_\theta(x').
$$
Under gradient flow on an empirical loss $\mathcal{L}(f(X),Y)$, the predictions on training inputs evolve as
$$
\frac{d}{dt}f_t(X)=-\Theta_{\theta_t}(X,X)\nabla_{f}\mathcal{L}(f_t(X),Y).
$$
This identity is exact for finite networks. The NTK limit is the statement that, under suitable infinite-width scaling, $\Theta_{\theta_t}$ converges to a deterministic kernel $\Theta_\infty$ and remains essentially constant over training. Then training becomes kernel gradient descent in function space.

[Jacot, Gabriel, and Hongler 2018](https://arxiv.org/abs/1806.07572) introduced the NTK and proved the central infinite-width picture for fully connected networks. [Lee et al. 2019](https://arxiv.org/abs/1902.06720) emphasized the linearized-network interpretation: in the wide limit, training $f_\theta$ is equivalent to training the first-order Taylor approximation
$$
f_{\theta_0}^{\mathrm{lin}}(x)=f_{\theta_0}(x)+\nabla_\theta f_{\theta_0}(x)\cdot(\theta-\theta_0).
$$
Thus the nonlinear parameterization remains present only through the tangent features at initialization.

For squared loss, the NTK limit gives an especially explicit formula. If $\Theta_\infty(X,X)$ is positive definite on the training inputs, then
$$
f_t(X)-Y=\exp(-t\Theta_\infty(X,X))(f_0(X)-Y),
$$
so the training error decays according to the eigenvalues of the limiting kernel matrix. In the infinite-time ridgeless limit, predictions are those of kernel interpolation with the NTK. This is why NTK theory connects overparameterized neural networks to classical kernel regression.

The word "lazy" captures the fact that parameters move little relative to their scale in the standard NTK regime. [Chizat, Oyallon, and Bach 2019](https://arxiv.org/abs/1812.07956) sharpen this point: lazy training is a parameterization-dependent regime, not a universal explanation of deep learning. A network can be very wide and still have different behavior if its output scale, learning-rate scale, or initialization scale is changed. [Woodworth et al. 2020](https://arxiv.org/abs/2002.09277) study the transition between kernel and rich regimes in overparameterized models, showing that the same finite architecture can be pushed toward a kernel-like or feature-learning description depending on scaling.

The NTK limit is powerful because it is analytically tractable. It gives convergence rates, generalization bounds through kernel methods, and a precise explanation of why very wide networks can fit data under gradient descent. Its limitation is equally important: when the kernel is frozen, hidden representations do not learn at leading order. This makes NTK theory a baseline theory of overparameterized training, not a complete theory of representation learning in modern deep networks.

### Takeaway

The NTK limit is a training limit in which the network behaves like its linearization around initialization. It explains lazy overparameterized optimization, but it deliberately removes leading-order feature learning.

## 4. Mean-Field Limits and Feature Learning

Mean-field theory starts from a different representation of the network. For a two-layer model, write
$$
f_{\rho}(x)=\int a\,\sigma(w\cdot x+b)\,d\rho(a,w,b),
$$
where $\rho$ is a probability distribution over neuron parameters. A finite network corresponds to an empirical measure
$$
\rho_m=\frac{1}{m}\sum_{i=1}^{m}\delta_{(a_i,w_i,b_i)}.
$$
Training the finite neurons moves the particles in parameter space. As $m\to\infty$, one may obtain a deterministic evolution equation for $\rho_t$ rather than a fixed kernel. In a common gradient-flow formulation,
$$
\partial_t \rho_t = \nabla_\theta\cdot\left(\rho_t\nabla_\theta \Psi(\theta;\rho_t)\right),
$$
where $\Psi$ is a potential depending on the data, the loss, and the current distribution through $f_{\rho_t}$.

The major conceptual difference from NTK theory is nonlinearity. In the NTK limit, the tangent kernel is fixed at initialization. In the mean-field limit, the distribution of neurons evolves, so the representation changes over time. This is why mean-field theory is often described as a feature-learning limit. It is also why the analysis is harder: the limiting dynamics are nonlinear in the law $\rho_t$.

[Mei, Montanari, and Nguyen 2018](https://arxiv.org/abs/1804.06561) give a mean-field view of two-layer networks and relate training to optimization over distributions of neurons. [Sirignano and Spiliopoulos 2018](https://arxiv.org/abs/1805.01053) prove law-of-large-numbers type mean-field results. [Rotskoff and Vanden-Eijnden 2018](https://arxiv.org/abs/1805.00915) connect trainability of wide networks to interacting-particle dynamics. [Chizat and Bach 2018](https://arxiv.org/abs/1805.09545) develop an optimal-transport gradient-flow perspective and clarify global convergence mechanisms for overparameterized models in measure space.

Mean-field limits also connect naturally to Barron-space and variation-norm viewpoints from approximation theory. A two-layer network is a finite signed measure over ridge features; its infinite-width counterpart is an integral representation. The mean-field training problem therefore optimizes over measures rather than finite parameter vectors. This gives an appealing continuous model of feature selection, although rigorous convergence and global optimality require assumptions that should be checked carefully in each theorem.

Deep mean-field theory is not just the two-layer story repeated. In a multilayer network, neurons in different layers interact through nested random features and cross-layer dependencies. [Sirignano and Spiliopoulos 2019](https://arxiv.org/abs/1903.04440) study mean-field analysis of deep networks. [Nguyen 2019](https://arxiv.org/abs/1902.02880) proposes formal multilayer mean-field dynamics, and [Nguyen and Pham 2020](https://arxiv.org/abs/2001.11443) develop a rigorous framework based on neuronal embeddings. These works show that deep mean-field limits can be meaningful, but the limiting object is more complicated than a single probability measure over independent two-layer neurons.

There is also a feature-learning infinite-width line that does not fit neatly into classical two-layer mean field. [Yang and Hu 2020](https://arxiv.org/abs/2011.14522) show that standard and NTK parameterizations do not give the desired feature-learning infinite-width behavior, and introduce maximal-update ideas that later support muP. [Bordelon and Pehlevan 2022](https://arxiv.org/abs/2205.09653) analyze infinite-width feature learning using a self-consistent dynamical field theory of kernel evolution. These papers shift the question from "does the kernel freeze?" to "which parameterization gives a stable nontrivial evolution of representations?"

### Takeaway

Mean-field limits describe networks as distributions of interacting neurons or features. They are central when the goal is to model representation learning rather than only lazy kernel training.

## 5. Parameterization, Tensor Programs, and muP

A finite architecture does not determine its scaling limit by itself. The same symbolic network can have different limits depending on whether weights are scaled like $1/\sqrt{m}$, outputs like $1/\sqrt{m}$ or $1/m$, learning rates like $1$, $m$, or $1/m$, and residual branches like $1$, $1/\sqrt{L}$, or $1/L$. Parameterization is therefore part of the theorem.

This is easy to see in a two-layer model. The NTK-style model
$$
f_m(x)=\frac{1}{\sqrt{m}}\sum_{i=1}^{m} a_i\sigma(w_i\cdot x)
$$
has random-output fluctuations of order one at initialization and often enters a lazy linearized regime at large width. A mean-field-style model
$$
f_m(x)=\frac{1}{m}\sum_{i=1}^{m} a_i\sigma(w_i\cdot x)
$$
with compatible initialization and learning-rate scaling instead behaves like an empirical measure approximation to an integral. The formulas look similar, but the large-width dynamics are different.

Tensor Programs give a systematic language for tracking such limits through complicated neural computations. [Yang 2019](https://arxiv.org/abs/1910.12478) proves architecture-general GP limits. [Yang 2020](https://arxiv.org/abs/2006.14548) extends the framework to NTK limits. The point is not only technical convenience; it is conceptual. Once networks include convolutions, attention, skip connections, layer normalization, and weight sharing, informal CLT reasoning can miss dependencies. Tensor Programs specify which random tensors become asymptotically Gaussian or non-Gaussian and how their covariances evolve.

The muP and muTransfer line asks a practical version of the same question: can a scaling parameterization make hyperparameters transfer across model sizes? [Yang and Hu 2020](https://arxiv.org/abs/2011.14522) identify parameterizations that preserve feature learning in infinite-width limits, and [Yang et al. 2022](https://arxiv.org/abs/2203.03466) show that in maximal-update parameterization many tuned hyperparameters can transfer from smaller to larger networks. This is a different use of scaling-limit theory from proving convergence to a kernel: the limit is used to design finite-model training rules whose behavior is stable across size.

For survey reading, parameterization should be recorded before the result. Many apparent disagreements in the literature are not contradictions; they are statements about different scalings. A theorem saying the NTK stays fixed, a theorem saying features move, and an experiment showing stable hyperparameter transfer can all be correct if they refer to different normalizations.

### Takeaway

Scaling-limit theory is partly a theory of parameterization. To understand a paper, record the weight scale, output scale, learning-rate scale, residual scale, and which dimensions are being sent to infinity.

## 6. Finite-Width Corrections

Infinite-width theorems are leading-order statements. A finite network is not equal to its limit; it fluctuates around it. Finite-width correction theory asks how large those fluctuations are, how they depend on depth and training time, and whether the corrections are only small noise or actually change the qualitative behavior.

At initialization, many wide-network observables have expansions of the form
$$
Q_m = Q_\infty + m^{-1/2}\xi + O(m^{-1}),
$$
where $\xi$ is a limiting Gaussian or non-Gaussian fluctuation depending on the observable. For training dynamics, corrections may involve both kernel fluctuations and the movement of the kernel during training. These terms matter when width is large but not astronomically large, which is the practical regime.

[Hanin and Nica 2019](https://arxiv.org/abs/1909.05989) show that finite depth and width corrections to the NTK can scale sharply with depth-to-width ratios. In particular, when depth and width grow together, the NTK need not behave like the deterministic fixed-depth infinite-width kernel. This is a recurring theme: fixed-depth infinite-width theory can be misleading for very deep networks unless depth dependence is controlled.

[Dyer and Gur-Ari 2019](https://arxiv.org/abs/1909.11304) introduce a Feynman-diagram approach for wide-network asymptotics. The value of this method is that it organizes higher-order terms systematically rather than treating all deviations from the infinite-width limit as unstructured error. [Pham and Nguyen 2021](https://proceedings.neurips.cc/paper/2021/hash/2639ba2137371773aa1e64e7735cdb30-Abstract.html) study limiting fluctuations for multilayer mean-field training, and [Bordelon and Pehlevan 2023](https://arxiv.org/abs/2304.03408) analyze finite-width fluctuations around dynamical mean-field descriptions of feature learning.

Finite-width corrections also clarify why infinite-width theories can be useful but incomplete. NTK theory may predict the leading trend of very wide training, while finite-width evolution explains representation drift, stochasticity across random seeds, and deviations that become important with depth. In practice, a good scaling-limit paper should state not only the limit but also the regime in which finite-size errors remain controlled.

### Takeaway

Finite-width theory studies the gap between the asymptotic model and real networks. It is essential when depth grows, when width is only moderately large, or when feature learning appears as a correction to a leading kernel limit.

## 7. Depth Limits and Signal Propagation

Depth introduces a different kind of scaling problem. Even at infinite width, repeatedly composing random layers can cause signals to explode, vanish, become perfectly correlated, or lose rank. The basic mean-field signal-propagation analysis tracks scalar statistics such as the variance of activations and the correlation between two inputs as depth grows.

For an MLP with activation $\phi$, a common recurrence has the form
$$
q^{\ell+1}=\sigma_b^2+\sigma_w^2\mathbb{E}[\phi(\sqrt{q^\ell}Z)^2],
$$
and for two inputs the correlation $c^\ell$ evolves by a map
$$
c^{\ell+1}=F(c^\ell;\sigma_w,\sigma_b,\phi).
$$
The derivative of this map near $c=1$ controls whether nearby signals separate or collapse with depth. The "edge of chaos" refers to critical parameter choices where signal propagation is neither rapidly ordered nor chaotic.

[Schoenholz et al. 2016](https://arxiv.org/abs/1611.01232) popularized this edge-of-chaos perspective for deep random networks and connected it to trainability. [Yang and Schoenholz 2017](https://arxiv.org/abs/1712.08969) extend signal-propagation ideas to residual networks, where skip connections change the correct depth scaling. [Xiao et al. 2018](https://arxiv.org/abs/1806.05393) analyze dynamical isometry in CNNs, focusing on the singular-value spectrum of input-output Jacobians rather than only scalar variances.

Residual networks motivate continuous-depth limits. If a residual block has the form
$$
h_{\ell+1}=h_\ell+\frac{1}{L}g_\ell(h_\ell),
$$
then as $L\to\infty$ one expects an ODE-like limit
$$
\frac{dh(t)}{dt}=g_t(h(t)).
$$
If the residual scale is instead $1/\sqrt{L}$ or the blocks are random, diffusion-like or rougher limits can appear. [Chen et al. 2018](https://arxiv.org/abs/1806.07366) introduced Neural ODEs as a modeling framework; the scaling-limit literature around ResNets studies when discrete residual architectures converge to deterministic or stochastic continuous-depth dynamics.

Transformers have their own signal-propagation issues. [Noci et al. 2022](https://arxiv.org/abs/2206.03126) show that stacking self-attention layers can cause rank collapse of token representations at initialization and can harm query/key gradients. This is a depth phenomenon, but it interacts with width, normalization, residual scaling, and context length. It illustrates why depth limits for modern architectures cannot be reduced to the scalar MLP recursions alone.

### Takeaway

Depth limits ask whether information and gradients survive repeated composition. The main objects are correlation maps, Jacobian spectra, residual scaling, continuous-depth dynamics, and architecture-specific collapse phenomena.

## 8. Joint Width-Depth and High-Dimensional Proportional Limits

Classical infinite-width theory usually fixes depth and data dimension. This makes the math cleaner, but it is not the only relevant asymptotic. If depth $L$ and width $m$ grow together, finite-width errors can accumulate across layers. A correction that is small for one layer may become order one after many layers.

[Li, Nica, and Roy 2021](https://arxiv.org/abs/2106.04013) study ResNets in an infinite-depth-and-width limit where depth and width tend to infinity with a fixed ratio. They show that the resulting initialization law can be log-Gaussian rather than the usual Gaussian-process limit. The lesson is not only about ResNets; it is that the order and relative rate of limits matter. Taking $m\to\infty$ first at fixed $L$, then $L\to\infty$, can yield a different answer from taking $m,L\to\infty$ together.

High-dimensional proportional limits scale data dimension, sample size, and number of features together. A typical regime is
$$
n,d,m\to\infty,\qquad \frac{n}{d}\to \gamma,\qquad \frac{m}{d}\to \psi.
$$
This is closer to random-matrix theory and statistical mechanics than to classical fixed-input kernel theory. It can produce precise formulas for test risk, double descent, and the effect of feature learning under structured data models.

[Mei and Montanari 2019](https://arxiv.org/abs/1908.05355) analyze random-features regression in high dimension and characterize generalization in proportional limits. [Ba et al. 2022](https://arxiv.org/abs/2205.01445) study how one gradient step on the first layer changes the representation in a proportional high-dimensional regime, showing that feature learning can appear immediately and can improve over fixed random features. This literature is especially useful for distinguishing kernel behavior from feature learning in settings where dimension and data size are both large.

These proportional theories use stronger data assumptions than architecture-only infinite-width theorems. They often work with teacher-student models, Gaussian covariates, single-index targets, or random-feature approximations. That is not a weakness; it is the price of obtaining precise risk formulas. The results answer a different question: not just "what function process does the network converge to?" but "what prediction risk does learning have when data, dimension, and width are all comparable?"

### Takeaway

Joint and proportional limits study regimes where classical fixed-depth, fixed-dimension infinite-width theory may fail. They are the natural bridge from neural-network asymptotics to random matrix theory and high-dimensional statistics.

## 9. Transformer Width, Head, and Depth Limits

Transformers add several scale parameters that are absent from MLPs. A single attention head computes
$$
\operatorname{Attn}(X)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+B\right)V,
$$
where $Q=XW_Q$, $K=XW_K$, $V=XW_V$, $d_k$ is the key/query dimension, and $B$ may encode causal masking or positional bias. The factor $1/\sqrt{d_k}$ from [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) is itself a scaling choice: without it, dot-product logits can grow with dimension and softmax can saturate.

In a transformer, sending the MLP hidden dimension to infinity is not the same as sending $d_k$ to infinity, the value dimension to infinity, the number of heads $H$ to infinity, or the depth $L$ to infinity. The softmax creates ratios of random sums, and the attention weights couple all tokens. This can lead to non-Gaussian behavior even when linear projections are high-dimensional and Gaussian at intermediate stages.

[Bordelon, Chaudhry, and Pehlevan 2024](https://arxiv.org/abs/2405.15712) analyze infinite limits of multi-head transformer dynamics in feature-learning regimes. They identify parameterizations with well-defined infinite width and depth limits and show that different limits, such as infinite key/query dimension, infinite heads, and infinite depth, lead to different dynamical mean-field descriptions. This is a useful modern example of why "make the transformer wide" is not a mathematically precise scaling statement.

[Sakai, Karakida, and Imaizumi 2025](https://arxiv.org/abs/2506.00846) study the infinite-width limit of a single attention layer using Tensor Programs and show that the realistic finite-head attention limit can be fundamentally non-Gaussian, with a hierarchical structure that is Gaussian only conditionally on random similarity scores. This result is a warning against assuming that all large-width transformer components reduce to ordinary GP recursions.

Transformer depth creates additional instability and collapse phenomena. [Noci et al. 2022](https://arxiv.org/abs/2206.03126) analyze rank collapse and signal propagation in transformers, showing how residual scaling and architectural hyperparameters affect token-rank preservation and gradient norms. In practical transformers, normalization placement, residual branch scale, attention temperature, and head dimension all interact; the scaling limit must specify them.

### Takeaway

Transformer scaling limits are multidimensional. Model dimension, head dimension, number of heads, MLP width, residual scale, normalization, and depth are separate asymptotic knobs, and different choices lead to different limiting theories.

## 10. Transformer Mean-Field and Attention Dynamics

Attention can also be studied as an interacting-particle system. At a fixed layer, the token representations $x_1,\ldots,x_T$ interact through attention weights
$$
A_{ij}=\frac{\exp(\beta\,q_i^\top k_j)}{\sum_{\ell=1}^{T}\exp(\beta\,q_i^\top k_\ell)},
$$
and the next representation of token $i$ depends on a weighted average of value vectors. When the number of tokens, feature dimension, heads, or layers becomes large, one can ask whether the empirical distribution of token states converges to a deterministic measure-valued dynamics.

This perspective is different from NTK theory. NTK asks how outputs change as parameters are trained. Transformer mean-field dynamics often ask how token representations evolve through attention layers or through simplified training dynamics. The limiting object may be a nonlinear transport equation, a gradient flow over probability measures, a dynamical mean-field system of order parameters, or a consensus/clustering process.

[Kim and Suzuki 2024](https://arxiv.org/abs/2402.01258) study nonconvex mean-field dynamics on the attention landscape and analyze how transformers can learn nonlinear features in context. [Poc-Lopez and Aguilera 2024](https://arxiv.org/abs/2406.07247) use dynamical mean-field methods for self-attention neural networks, connecting simplified attention models to nonequilibrium statistical mechanics and phase transitions. [Burger et al. 2025](https://arxiv.org/abs/2501.03096) analyze mean-field models arising from self-attention dynamics with layer normalization and formulate special cases as gradient flows on probability measures over the sphere.

[Rigollet 2025](https://arxiv.org/abs/2512.01868) develops a mean-field framework for transformer attention as interacting particles on the sphere. In this view, attention can produce clustering, metastability, consensus, and phase transitions depending on normalization and scaling. This is directly relevant to long-depth and long-context behavior because a failure mode of repeated attention is that token representations collapse toward a small number of directions.

The useful synthesis is that attention is both a feature-selection operation and an interaction mechanism. Kernel limits describe attention through fixed similarity statistics; mean-field limits describe how the distribution of token states itself changes under repeated interactions. Both are simplifications, but they emphasize different mechanisms.

### Takeaway

Transformer mean-field theory treats tokens or hidden states as interacting particles. It is a natural language for studying clustering, consensus, rank collapse, and phase transitions in attention dynamics.

## 11. Long-Context and Sequence-Length Limits

The long-context limit sends the sequence length $T$ to infinity. This should be separated from ordinary width limits. A model can be infinitely wide at a fixed context length and still fail when $T$ grows. Conversely, a long-context theorem may hold for a simplified finite-dimensional attention model without saying anything about infinite-width training.

The attention weights are the central object:
$$
A_{ij}^{(T)}=\frac{\exp(\beta_T s_{ij}+b_{ij}^{(T)})}{\sum_{\ell=1}^{T}\exp(\beta_T s_{i\ell}+b_{i\ell}^{(T)})}.
$$
Here $s_{ij}$ is a content similarity score, $b_{ij}^{(T)}$ may come from a positional encoding or causal mask, and $\beta_T$ is a possible context-length-dependent attention scale. As $T$ grows, the denominator contains more competing tokens. If scores are not scaled appropriately, attention can become too diffuse, too concentrated on extremes, or biased by positional terms in a way that was not visible at the training length.

Length extrapolation studies ask whether a model trained on shorter sequences performs on longer sequences. Positional encoding is often the first bottleneck. [Press, Smith, and Lewis 2021](https://arxiv.org/abs/2108.12409) propose ALiBi, which adds linear attention biases and was motivated by train-short-test-long extrapolation. [Su et al. 2021](https://arxiv.org/abs/2104.09864) introduce RoPE, which encodes relative position through rotations of query and key coordinates. [Kazemnejad et al. 2023](https://arxiv.org/abs/2305.19466) systematically compare positional encodings for length generalization, and [Huang et al. 2024](https://arxiv.org/abs/2410.02140) provide a formal framework for analyzing when length generalization is identifiable in causal transformers with positional encodings.

RoPE-specific long-context theory studies how the rotary base and frequency spectrum constrain extrapolation. [Men et al. 2024](https://arxiv.org/abs/2405.14591) argue that the RoPE base bounds context length through long-term decay properties. [Liu 2026](https://arxiv.org/abs/2602.10959) interprets RoPE as phase modulation and derives theoretical bounds on the RoPE base for preserving positional coherence over long contexts. These papers are useful because they turn the vague statement "increase the context window" into concrete frequency and stability constraints.

A second branch studies context-length-induced attention pathologies. As $T$ grows, a softmax over many nearly comparable scores can become diffuse, causing non-informative tokens to receive nontrivial total mass. In other regimes, extreme-value effects can make attention focus on accidental maxima. Deep stacks can also push token representations toward low-rank or clustered states. [Noci et al. 2022](https://arxiv.org/abs/2206.03126) study rank collapse from stacked attention layers, while [Rigollet 2025](https://arxiv.org/abs/2512.01868) and [Chen, Lin, Polyanskiy, and Rigollet 2026](https://openreview.net/forum?id=7SLtElfqCW) connect long-context attention scaling to clustering and phase transitions.

The 2026 critical-attention-scaling result is a clean example of a true sequence-length scaling law. In a simplified model, attention behavior changes qualitatively depending on the length-dependent scale $\beta_T$: insufficient scaling collapses tokens toward a single direction, excessive scaling makes attention too close to identity, and a critical logarithmic scale $\beta_T\asymp \log T$ maintains sparse content-adaptive interactions. The point is not that every production transformer is exactly described by that model; the point is that context length creates its own asymptotic temperature problem.

Long-context theory should also be separated from computational complexity. Sparse, linear, recurrent, or memory-augmented attention mechanisms reduce the cost of processing long sequences, but a cheaper mechanism is not automatically a good long-context limit. The mathematical question here is whether the attention distribution, positional information, and representation geometry remain meaningful as $T$ grows.

### Takeaway

The long-context limit is a sequence-length asymptotic. It studies positional extrapolation, attention temperature, entropy or dispersion of attention weights, rank collapse, localization, and phase transitions as the number of tokens grows.

## 12. How to Read the Literature

When reading a scaling-limit paper, first identify the asymptotic variables. A paper may take width $m\to\infty$, depth $L\to\infty$, sample size $n\to\infty$, input dimension $d\to\infty$, head count $H\to\infty$, key/query dimension $d_k\to\infty$, or context length $T\to\infty$. These are not interchangeable.

Second, record the parameterization. The relevant details are weight variance, output normalization, learning-rate scaling, residual-branch scaling, attention-logit scaling, and normalization placement. Many qualitative claims, such as "the kernel is fixed" or "features learn," are only true under a specific scaling.

Third, separate initialization, training, and inference. NNGP theory is mostly an initialization or Bayesian-prior theory. NTK theory is a gradient-flow training theory in a lazy regime. Mean-field theory is a feature-learning training theory. Signal-propagation theory is often an initialization-through-depth theory. Long-context theory may be about inference at longer sequence lengths even when training length was finite.

Fourth, track the limiting object. Gaussian processes, deterministic kernels, Wasserstein gradient flows, ODEs, SDEs, random-matrix risk formulas, tensor-program recursions, and interacting-particle systems answer different questions. They should not be evaluated as if they were competing approximations to the same object.

Finally, ask what the limit leaves out. Fixed-kernel limits often leave out representation learning. Mean-field limits may rely on idealized distributions and continuous-time training. High-dimensional proportional limits often rely on teacher-student assumptions. Transformer long-context limits often simplify attention to isolate one mechanism. These simplifications are useful when they are read as controlled lenses, not as complete descriptions of every finite model.

### Overall Takeaway

The original three topics, the neural tangent kernel, the mean-field limit, and the long-context transformer limit, are three major branches of a broader scaling-limit taxonomy. The clean map is: NNGP describes random functions at infinite width; NTK describes lazy training as kernel dynamics; mean-field and muP-style limits describe feature learning; finite-width and depth limits describe corrections and signal propagation; proportional limits connect to high-dimensional statistics; and transformer-specific limits split further by width, heads, depth, token interactions, and context length.
