# Expressiveness of MLP and CNN

## Outline

As discussed in the basic machine learning section, the learning problem starts from an unknown ground-truth function $f$. We choose a parametric model family, or ansatz $\{f_\theta : \theta \in \Theta\}$, collect data $\{(x_i, y_i)\}_{i=1}^n$ related to $f$ such that $y_i = f(x_i)$, and then use the data to select a parameter $\theta$. Expressiveness asks a question that comes before optimization and generalization: can some $f_\theta$ in the chosen family represent or approximate the ground truth $f$ at all?

1. Notions of expressiveness

   First we clarify what it means for an architecture to be expressive. The most direct question is whether the model family can approximate any ground truth $f$. A stronger question is whether it can approximate every function in a smaller target class $\mathcal{F}$, and how many parameters are needed to achieve a given accuracy. This section also discusses how to design simple controlled experiments for testing expressiveness: choose a known target family, match parameter budgets, and measure approximation error against the known ground truth.

2. Universal approximation

   If we do not know anything about the ground truth, the natural question is whether the architecture can approximate any function in a large ambient function space. This is the universal approximation problem. Universal approximation is a qualitative baseline: it tells us that the model family is not missing whole classes of targets, but it usually does not tell us whether the required number of parameters is reasonable.

3. Expressiveness of MLPs

   Universal approximation for MLPs can require exponentially many parameters when the target is an arbitrary high-dimensional function. Therefore the next question is whether MLPs become efficient when the ground truth has additional structure. A central example is compositional structure: if $f$ is built from simpler lower-dimensional functions, can a deep MLP represent or approximate it with far fewer parameters than a shallow model? Even without assuming that the target function itself has a known compositional structure, the Kolmogorov-Arnold theorem shows that continuous multivariate functions can be represented through compositions of univariate functions, giving a different perspective on the expressive power of composition.

4. More on expressiveness of MLPs

   After the basic MLP results, we study depth, width, and parameter tradeoffs. The same function may be easy for a deep network but expensive for a shallow one, or easy with enough width but impossible below a width threshold. Special test cases such as step functions, Boolean functions, and finite-sample memorization help make these questions concrete. They should be separated because approximation on continuous domains, exact representation on $\{0,1\}^d$, and interpolation of finite datasets are different problems.

5. Expressiveness of CNNs

   CNNs are motivated by structured ground truths. Locality means the target depends mainly on nearby variables or local patterns. Invariance means the output should not change under certain transformations, while equivariance means the output transforms predictably when the input is transformed. If the ground truth has locality, translation structure, invariance, or equivariance, CNNs may approximate it with fewer parameters than fully connected networks.

6. More on expressiveness of CNNs

   To understand what CNNs really gain, we should analyze simple image-inspired
   function classes rather than only arbitrary functions. The useful toy models
   are functions of patches, functions that detect a local feature appearing at
   any location, local compositional functions, and translation-invariant or
   shift-equivariant maps. Recent theory studies such classes directly, for
   example by comparing CNNs, locally connected networks, and fully connected
   networks on patch-based image tasks. These results help isolate when the gain
   comes from locality, when it comes from weight sharing, and when it comes from
   pooling or equivariant structure.

## 1. Notions of Expressiveness

Expressiveness asks whether the chosen ansatz can represent or approximate the
ground truth. The basic object is the approximation error
$$
\inf_{\theta \in \Theta} d(f_\theta, f),
$$
where $d$ is a task-appropriate metric, such as $L^2$, $L^\infty$,
classification error, or empirical error on samples.

For a single target $f$, expressiveness means that this error can be made
small. For a target family $\mathcal{F}$, expressiveness asks whether every
$f \in \mathcal{F}$ can be approximated and how many parameters are needed:
$$
\forall f \in \mathcal{F}, \quad \exists f_\theta
\quad \text{such that} \quad d(f_\theta, f) \leq \epsilon.
$$
The parameter count needed as a function of $\epsilon$, dimension, and the
structure of $\mathcal{F}$ is the main quantitative measure.

There are three useful levels:

- Exact representation: some $f_\theta$ equals $f$ on the relevant domain.
- Approximation: some $f_\theta$ is within error $\epsilon$ of $f$.
- Efficient approximation: the required depth, width, or parameter count grows
  slowly with dimension and $1/\epsilon$.

To test expressiveness experimentally, choose a known target family, generate
data from it, train several architectures under matched parameter budgets, and
measure error against the known ground truth rather than only training loss.
Varying dimension, smoothness, locality, or compositional depth reveals how the
required model size scales.

## 2. Universal Approximation

If we do not know anything about the ground truth, the broadest question is whether an architecture can approximate every function in a large function space. For MLPs this usually means approximating every continuous function on a compact subset of $\mathbb{R}^d$, under a norm such as $L^\infty$.

### A Standard Universal Approximation Theorem

Let $K \subset \mathbb{R}^d$ be compact. Let $\sigma:\mathbb{R}\to\mathbb{R}$ be continuous and non-polynomial. Consider the one-hidden-layer network class
$$
\mathcal{N}_\sigma
= \left\{
x \mapsto \sum_{j=1}^m a_j \sigma(w_j \cdot x + b_j)
:\; m \in \mathbb{N},\; a_j,b_j \in \mathbb{R},\; w_j \in \mathbb{R}^d
\right\}.
$$
Then $\mathcal{N}_\sigma$ is dense in $C(K)$ with respect to the uniform norm: for every $f \in C(K)$ and every $\epsilon>0$, there exists $g \in \mathcal{N}_\sigma$ such that
$$
\sup_{x\in K} |f(x)-g(x)| \leq \epsilon.
$$

This is the simplest modern form of the universal approximation theorem. Older versions assumed a sigmoidal activation such as the logistic function; see [Cybenko 1989](https://doi.org/10.1007/BF02551274). The non-polynomial condition is the standard sharp condition for continuous activations; see [Leshno, Lin, Pinkus, and Schocken 1993](https://doi.org/10.1016/S0893-6080(05)80131-5). If $\sigma$ is a polynomial, then a one-hidden-layer network with activation $\sigma$ can only produce polynomials of bounded degree determined by $\sigma$, so it cannot be dense in $C(K)$.

### Proof Sketch

The proof is usually done by contradiction and uses a separation theorem from functional analysis. Let
$$
\mathcal{M}
= \operatorname{span}\{\sigma(w\cdot x+b): w\in\mathbb{R}^d,\ b\in\mathbb{R}\}
\subset C(K).
$$
We want to show that the uniform closure of $\mathcal{M}$ is all of $C(K)$. Suppose not. By the Hahn-Banach theorem, there exists a nonzero continuous linear functional $L$ on $C(K)$ such that $L(g)=0$ for all $g\in\mathcal{M}$. By the Riesz representation theorem, this functional has the form
$$
L(g)=\int_K g(x)\,d\mu(x)
$$
for some nonzero finite signed Borel measure $\mu$ on $K$. Therefore
$$
\int_K \sigma(w\cdot x+b)\,d\mu(x)=0
\quad
\text{for all } w\in\mathbb{R}^d,\ b\in\mathbb{R}.
$$

The key mathematical fact is that a non-polynomial activation has enough one-dimensional shifts and scalings to separate signed measures. For smooth activations, the intuition is that differentiating $\int_K \sigma(w\cdot x+b)\,d\mu(x)=0$ with respect to $w$ at $w=0$ produces equations involving all moments $\int_K x^\alpha\,d\mu(x)$, and a non-polynomial $\sigma$ supplies enough nonzero derivative orders to force all those moments, hence $\mu$, to vanish. In other words, if the above integral is zero for every ridge function $x\mapsto \sigma(w\cdot x+b)$, then $\mu$ must be the zero measure. This contradicts the choice of a nonzero $\mu$. Hence no nonzero continuous linear functional can annihilate the closure of $\mathcal{M}$, so the closure must be all of $C(K)$.

For sigmoidal activations, this separating-measures step is often proved using limits. A large rescaling of $\sigma(w\cdot x+b)$ approaches an indicator of a half-space. If a measure integrates to zero over all such soft half-space indicators, then it assigns zero mass to all half-spaces, and hence must be the zero measure. For general non-polynomial continuous activations, the proof uses a more refined density result for ridge functions.

The important conceptual point is that universal approximation is not proved by constructing the exact network we would train in practice. It is an existence argument: finite linear combinations of ridge functions are rich enough that no nonzero measure can be orthogonal to all of them.

### What Universal Approximation Does and Does Not Say

Universal approximation is a qualitative baseline. It says that the model class is not fundamentally missing functions in the chosen target space. It does not say that the approximation is efficient, stable, learnable, or easy to find by gradient descent.

In high dimension, approximating an arbitrary continuous function can require exponentially many parameters. This is not a defect of the theorem; it is the curse of dimensionality. The theorem says that approximation is possible after allowing the width $m$ to grow, but it gives no useful guarantee that $m$ grows polynomially in $d$ or $1/\epsilon$ for an arbitrary target.

When reading a universal approximation theorem, record:

- the function space being approximated;
- the domain;
- the activation function;
- the norm or topology;
- whether depth, width, or both are allowed to grow.

### Generalizations

There are many universal approximation theorems. They differ mainly in the target function space, the activation assumptions, the architecture class, and the topology of approximation.

**Activation and target-space variants.** Universal approximation depends both on the activation and on the topology of approximation. [Cybenko 1989](https://doi.org/10.1007/BF02551274) covers continuous sigmoidal activations, [Hornik, Stinchcombe, and White 1989](https://doi.org/10.1016/0893-6080(89)90020-8) treats broad feedforward-network approximation, and [Leshno, Lin, Pinkus, and Schocken 1993](https://doi.org/10.1016/S0893-6080(05)80131-5) gives the standard one-hidden-layer characterization: roughly, a locally bounded piecewise continuous activation gives density in $C(K)$ exactly when it is not an algebraic polynomial almost everywhere.

- Common activations: sigmoid and tanh are covered by Cybenko's theorem because they are continuous sigmoidal functions; ReLU, leaky ReLU, ELU, and GELU are covered by the Leshno-Lin-Pinkus-Schocken characterization because each is locally bounded, piecewise continuous or continuous, and not equal almost everywhere to any algebraic polynomial. ReLU and leaky ReLU are piecewise linear but not a single global polynomial because the slope changes at $0$; ELU is piecewise-defined with an exponential branch; GELU contains the Gaussian CDF factor, so it is smooth but not polynomial.
- $L^p$: Instead of uniform approximation on $C(K)$, one can ask for density in $L^p(K)$ for $1\leq p<\infty$. This is often easier because $L^p$ ignores sets of measure zero, but it is weaker than uniform approximation because it does not control the pointwise worst-case error.
- Sobolev/Besov/Holder: Many theorems quantify how ReLU or piecewise-polynomial networks approximate functions in smoothness classes. [Yarotsky 2017](https://doi.org/10.1016/j.neunet.2017.07.002) is a canonical Sobolev-style reference for deep ReLU approximation, and [Petersen and Voigtlaender 2018](https://doi.org/10.1016/j.neunet.2018.08.019) studies optimal approximation of piecewise smooth functions, including discontinuities across smooth hypersurfaces.
- Non-continuous targets: ReLU, sigmoid, and tanh networks are continuous functions of the input, so they cannot uniformly approximate a discontinuous function on a compact domain. They can approximate discontinuous functions in $L^p$, or uniformly away from the discontinuity set; this matters for classification, where the label function may be discontinuous but the score or probability function is often modeled continuously.


**Vector-valued functions.** For functions $f:K\to\mathbb{R}^q$, universality usually follows by approximating each coordinate separately and stacking the outputs. The deep narrow network result of [Kidger and Lyons 2020](https://proceedings.mlr.press/v125/kidger20a.html) states a vector-valued version directly: for input dimension $n$ and output dimension $m$, width $n+m+2$ is enough under mild nonaffine activation assumptions. The parameter count may still scale at least linearly with $q$ unless the coordinates share structure.

**Approximation rates and parameter counts.** Qualitative universality does not give rates. Quantitative approximation theory asks for the number of parameters $N(\epsilon,\mathcal{F})$ needed so that every $f$ in a target class $\mathcal{F}$ can be approximated to error $\epsilon$. For generic $s$-smooth functions on $[0,1]^d$, rates typically scale like
$$
N(\epsilon,\mathcal{F}) \approx \epsilon^{-d/s}
$$
up to architecture- and norm-dependent details. This exponential dependence on $d$ is the curse of dimensionality. [Yarotsky 2017](https://doi.org/10.1016/j.neunet.2017.07.002) and [Yarotsky 2018](https://proceedings.mlr.press/v75/yarotsky18a.html) are representative ReLU approximation-rate papers with matching or near-matching upper and lower bounds. Better rates require additional structure, such as low intrinsic dimension, sparsity, compositional form, locality, or symmetry.

### Takeaway

The universal approximation theorem justifies MLPs as a very general ansatz: with a standard non-polynomial activation, even one hidden layer can approximate any continuous function on a compact domain. But universality alone is a weak guarantee. For deep learning, the main question is quantitative: which functions can be approximated with a reasonable number of parameters, and which architectural biases reduce the dependence on dimension?

## 3. Expressiveness of MLPs

For MLPs, the first question is classical universality: under what conditions on
the activation function and architecture can an MLP approximate arbitrary
continuous functions? This explains why MLPs are a reasonable general ansatz,
but it does not explain when they are parameter-efficient.

The next question is whether MLPs exploit structure in the ground truth. If
$f$ is compositional, hierarchical, sparse, smooth, or low-dimensional in some
sense, an MLP may approximate it with fewer parameters than would be needed for
an arbitrary function. Depth is especially relevant for compositional structure:
a deep network can mirror the nested structure of the target.

There is also a separate composition viewpoint from the Kolmogorov-Arnold
theorem. Even without assuming that the target comes with a known compositional
structure, continuous multivariate functions admit representations using
compositions of univariate functions. This gives a conceptual reason to study
composition as a source of expressiveness, though it does not by itself settle
practical approximation rates for standard neural networks.

Main MLP questions:

- Which activation functions give universality?
- What target classes admit efficient approximation?
- How do approximation rates depend on dimension and smoothness?
- When does depth reduce the number of required parameters?

## 4. More on Expressiveness of MLPs

Depth, width, and parameter count should be treated as different resources. A
function may be easy for a deep network but expensive for a shallow one; another
function may require a minimum width before approximation is possible. Depth
separation results formalize this by giving functions that deep networks
represent or approximate with far fewer parameters than shallow networks.

Special test cases make these issues concrete:

- Step and indicator functions test approximation of discontinuities or sharp
  decision boundaries on continuous domains. Since ReLU networks are continuous,
  discontinuous step functions are usually approximation targets, not exact
  representation targets, on $\mathbb{R}^d$.
- Boolean functions on $\{0,1\}^d$ are finite-domain problems. Exact
  representation is possible in ways that are different from continuous-domain
  approximation, and the topic connects to threshold circuits.
- Finite-sample memorization asks whether a network can fit arbitrary labels
  $(x_i, y_i)$. This measures interpolation capacity, but it does not imply
  good approximation away from the sample points.

Useful complexity measures include depth, width, number of neurons, number of
nonzero parameters, weight magnitude, and number of linear regions. These should
not be conflated.

### TODO: Approximation Rates and MLP Resource Tradeoffs

**Fixed-depth versus growing-depth networks.** The theorem above already shows that depth two, meaning one hidden layer plus a linear output layer, is enough for qualitative universality. Deeper networks are not needed for mere density in $C(K)$. Depth matters for efficiency: [Yarotsky 2017](https://doi.org/10.1016/j.neunet.2017.07.002) gives upper and lower approximation bounds for deep ReLU networks on Sobolev-type smoothness classes, while [Telgarsky 2016](https://proceedings.mlr.press/v49/telgarsky16.html) gives depth-separation examples where deeper networks represent oscillatory functions much more efficiently than shallow networks.

**Minimal width results.** For ReLU networks on compact subsets of $\mathbb{R}^d$, depth can grow while width is fixed. [Lu, Pu, Wang, Hu, and Wang 2017](https://arxiv.org/abs/1709.02540) studies universality from the viewpoint of width, and [Hanin and Sellke 2017](https://arxiv.org/abs/1710.11278) shows the sharp scalar ReLU threshold: width $d+1$ is enough for universal approximation of continuous functions on compact domains, while width at most $d$ is not enough in general. Thus universality can be achieved by either making a shallow network very wide or making a narrow network sufficiently deep.

## 5. Expressiveness of CNNs

CNNs are designed for ground truths with spatial structure. The key question is
not only whether CNNs are universal, but whether they approximate structured
functions more efficiently than fully connected networks.

The relevant structures are:

- Locality: the output depends on nearby variables or local patterns.
- Weight sharing: the same local rule is reused across positions.
- Invariance: transforming the input should not change the output.
- Equivariance: transforming the input should transform the output in a
  predictable way.

For example, image classification often benefits from approximate translation invariance, while segmentation benefits from translation equivariance. A CNN can save parameters when the same local pattern matters at many positions, because convolution reuses weights rather than learning separate parameters for each location.

Main CNN questions:

- Under what conditions are CNNs universal?
- Which structured target classes do CNNs approximate efficiently?
- Which gains come from locality and which come from weight sharing?
- How much depth is needed for local information to combine into global
  information?

## 6. More on Expressiveness of CNNs

To understand CNN expressiveness, it is useful to study simple function classes inspired by pictures. The goal is not to model natural images perfectly. The goal is to isolate the structures that make CNNs useful: local patches, repeated features, spatial composition, and translation symmetry.

### Patch-Based Image Functions

A simple image model divides the input into $k$ patches, each of dimension
$d$. The label may depend on whether a signal or feature appears in one of the patches. This captures two basic properties of image tasks:

- locality: the relevant signal is contained in a small patch;
- translation structure: the same signal may appear at many possible locations.

One relevant result is the Dynamic Signal Distribution task in Lahoti, Karp, Winston, Singh, and Li. The task models an image as $k$ patches with a sparse signal that can appear in any patch. Their ICLR 2024 result gives a sample complexity separation: CNNs need about $\tilde O(k+d)$ samples, while locally connected networks need $\Omega(kd)$ samples; they also separate locally connected networks from fully connected networks. This is a clean theoretical model for the benefits of weight sharing and locality in image-like tasks. See [Lahoti et al. 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/71b17f00017da0d73823ccf7fbce2d4f-Abstract-Conference.html).

### Local Compositional Functions

Another simple class consists of functions built by repeatedly combining nearby features. This resembles the way CNN layers combine local edges or textures into larger patterns. In this setting, depth is important because each layer expands the effective receptive field and combines local information into more global features.

Approximation results for CNNs show that convolutional architectures can be advantageous for target classes with compositional form. Bao, Li, Shen, Tai, Wu, and Xiang analyze CNN approximation and show that, for certain compositional target classes, CNNs can be more parameter-efficient than fully connected networks. See [Bao et al. 2023](https://global-sci.org/index.php/EAJAM/article/view/9619).

### Translation-Invariant and Shift-Equivariant Functions

For classification, the desired output is often approximately invariant to
translation: moving an object should not change the class. For dense prediction tasks such as segmentation, the desired map is often shift-equivariant: shifting the input should shift the output.

This distinction matters. Convolutions are naturally translation-equivariant, not translation-invariant. Invariance usually requires additional operations such as pooling, global aggregation, or training over translated examples. Empirical work also shows that standard CNN classifiers are not automatically translation invariant in all settings, although they can learn invariance from suitable data. See [Biscione and Bowers 2021](https://www.jmlr.org/papers/v22/21-0019.html).

There are also approximation results for invariant function classes. For example, Li, Lin, and Shen study approximation of permutation-invariant functions, including translation-invariant functions relevant to image tasks, using symmetry-constrained deep architectures. See [Li, Lin, and Shen 2024](https://www.jmlr.org/papers/v25/22-0982.html).

### Feature-Detection Models

A picture-inspired expressiveness question is whether a CNN can implement
feature extraction: detect whether a local pattern appears somewhere in the
image and then classify based on the detected feature. This is the simplest
abstraction of many visual tasks.

Some recent preprints study mathematical models of image classification based on feature extraction and construct CNNs that realize such feature-detection functions. These are useful as toy models, even when they are not yet canonical theorems. See the OpenReview preprint [Revisiting the expressiveness of CNNs](https://openreview.net/forum?id=BDUB1wWR1X).

### What to Take From These Simple Classes

These examples suggest a better way to organize CNN expressiveness:

- Use patch tasks to test locality and weight sharing.
- Use local compositional functions to test the role of depth.
- Use translation-invariant functions to test classification-style symmetry.
- Use shift-equivariant maps to test dense-prediction-style symmetry.
- Compare CNNs with locally connected networks and fully connected networks to separate the effects of locality and sharing.

This is closer to the real picture than asking only whether CNNs are universal. The main question is: for which image-inspired ground-truth classes do CNNs achieve better parameter or sample complexity than less structured networks?

### TODO: CNN Universality and Symmetry-Constrained Universality

**CNNs and structured architectures.** CNNs can also be universal, but the statement depends on padding, channel count, pooling, boundary handling, and whether fully connected layers are allowed at the end. [Zhou 2020](https://doi.org/10.1016/j.acha.2019.06.004) proves a universality theorem for deep CNNs, showing that depth can compensate for convolutional weight sharing in a qualitative approximation sense. The more interesting question is not universality itself, but whether the CNN approximates translation-structured or local functions with fewer parameters than an unconstrained MLP.

**Invariant and equivariant networks.** If the target is known to be invariant or equivariant under a group action, then the relevant universal approximation theorem should be restricted to that function class. [Zaheer et al. 2017](https://papers.nips.cc/paper/6931-deep-sets) characterizes permutation-invariant set functions and motivates the Deep Sets architecture, while [Keriven and Peyre 2019](https://arxiv.org/abs/1905.04943) proves universality results for invariant and equivariant graph networks. These theorems are more useful than unrestricted universality when the ground truth has symmetry.

**Residual networks and other modern architectures.** ResNets, neural ODE models, graph neural networks, and transformers each have their own universality results under suitable assumptions. For example, [Yun et al. 2020](https://arxiv.org/abs/1912.10077) proves universal approximation results for transformers as sequence-to-sequence models. The same lesson applies: qualitative density is usually only the first test. The central expressiveness question is whether the architecture gives an efficient representation of the target family we actually care about.
