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

For MLPs, classical universality answers only the question of density: with a suitable non-polynomial activation, can the network approximate arbitrary continuous functions on compact domains? Section 2 says yes. The expressiveness question in this section is sharper: for which target classes can an MLP approximate with a reasonable number of units, layers, and parameters?

### From Universality to Efficient Approximation

The useful quantity is an approximation rate. For a target class $\mathcal{F}$ and a network budget $N$, define
$$
E_N(\mathcal{F})
= \sup_{f\in\mathcal{F}} \inf_{g\in\mathcal{N}_N} \|f-g\|,
$$
where $\mathcal{N}_N$ is a class of networks with at most $N$ parameters, neurons, or nonzero weights. Universal approximation says only that $E_N(\{f\})\to 0$ for each fixed continuous $f$ as $N\to\infty$. Efficient approximation asks how fast $E_N(\mathcal{F})$ decays uniformly over a structured class $\mathcal{F}$.

For generic $s$-smooth functions on $[0,1]^d$, the usual nonparametric approximation scale is roughly $\epsilon^{-d/s}$ parameters for accuracy $\epsilon$, up to constants and architecture-dependent details. This is the curse of dimensionality. Results such as [Yarotsky 2017](https://doi.org/10.1016/j.neunet.2017.07.002) show that deep ReLU networks achieve near-optimal rates for Sobolev-type smoothness classes, but the dimension dependence remains unless the target has more structure. The main role of MLP expressiveness theory is therefore to identify structures that reduce this dependence.

### Smoothness, Barron Structure, and Spectral Control

One classical positive result is Barron's theorem. [Barron 1993](https://doi.org/10.1109/18.256500) studies functions whose Fourier transform has a bounded first moment and shows that shallow networks can achieve dimension-free-looking $L^2$ approximation rates of order $O(1/n)$ in the squared error with $n$ hidden units, under the relevant Barron norm. The important point is not that all smooth functions are easy; the point is that spectral structure can make a high-dimensional function easier than a worst-case $C^s$ or Sobolev function.

Modern work often phrases this through Barron spaces or related variation norms. These classes are naturally matched to two-layer or mean-field neural networks: the target is represented as an integral over ridge features, and a finite network is a finite-sample approximation of that integral. This gives a different route to avoiding the curse of dimensionality than depth: instead of assuming local smoothness in all $d$ coordinates, one assumes a global ridge-feature representation with controlled norm. The survey [DeVore, Hanin, and Petrova 2021](https://doi.org/10.1017/S0962492921000052) is a broad reference for these approximation-theoretic viewpoints.

### Compositional and Hierarchical Structure

Depth becomes powerful when the target function is compositional. A typical model is a binary tree
$$
f(x_1,\ldots,x_d)
= h_{1}\bigl(h_{2}(x_{i_1},x_{i_2}), h_{3}(x_{i_3},x_{i_4}),\ldots\bigr),
$$
where each constituent function depends on only a few variables. A deep MLP can mirror this computation graph: early layers approximate low-dimensional components, later layers compose them, and the number of required units grows with the complexity of the small constituent functions rather than with the full ambient dimension $d$.

This is the core message of the compositional-function literature. [Poggio, Mhaskar, Rosasco, Miranda, and Liao 2017](https://doi.org/10.1007/s11633-017-1054-2) review results showing when deep networks can avoid the curse of dimensionality for hierarchical compositional functions, while shallow networks may require exponentially many units. The phrase "deep is better than shallow" should be read carefully: depth helps when the target has a compositional structure that the network can exploit. It does not mean that depth automatically defeats high dimensionality for arbitrary functions.

### Depth Separation

Depth separation results make the previous intuition formal by constructing functions that are easy for deeper networks and hard for shallower ones. [Eldan and Shamir 2016](https://proceedings.mlr.press/v49/eldan16.html) construct a radial function representable by a small three-layer network but requiring exponential width for approximation by two-layer networks under broad activation assumptions. [Telgarsky 2016](https://proceedings.mlr.press/v49/telgarsky16.html) gives another influential construction based on highly oscillatory sawtooth-like functions, showing that depth can create repeated folding/composition that shallow networks cannot efficiently reproduce.

These theorems are separator results, not universal claims about every practical target. Their value is conceptual: they prove that depth is a genuine expressiveness resource, not merely a different parameterization of width. They also explain why compositional toy functions are useful tests for expressiveness experiments: if the target is generated by repeated composition, a deep network can reuse a small module many times through layers.

### Linear Regions and Piecewise-Linear Complexity

For ReLU and leaky-ReLU MLPs, another concrete measure of expressiveness is the number of linear regions into which the network partitions input space. A shallow ReLU network forms a piecewise-linear function with regions induced by hyperplane arrangements. A deep ReLU network composes such partitions layer by layer, so later layers can fold and reuse regions created earlier.

[Montufar, Pascanu, Cho, and Bengio 2014](https://papers.nips.cc/paper/5422-on-the-number-of-linear-regions-of-deep-neural-networks) show that the number of linear regions can grow rapidly, often exponentially with depth for fixed width regimes. This does not by itself prove good approximation of a specific target, because many linear regions may be placed in unhelpful locations. But it gives a useful geometric explanation of why deep piecewise-linear networks can represent complicated decision boundaries and highly oscillatory functions with relatively few parameters.

### Kolmogorov-Arnold Viewpoint

There is also a separate composition viewpoint from the Kolmogorov-Arnold representation theorem. The theorem was originally part of the resolution of Hilbert's thirteenth problem: it showed that arbitrary continuous multivariate functions can be written using only univariate continuous functions and addition, so multivariate continuity does not by itself force irreducibly multivariate building blocks. See [Kolmogorov 1957](https://www.mathnet.ru/eng/dan22050) for the original superposition theorem; refinements appear in Sprecher's work, such as [Sprecher 1965](https://www.ams.org/journals/tran/1965-115-00/S0002-9947-1965-0210852-X/S0002-9947-1965-0210852-X.pdf).

A common sample statement is the following. For each $d\geq 2$, there exist continuous inner functions $\phi_{q,p}:[0,1]\to\mathbb{R}$, independent of the target function $f$, such that every $f\in C([0,1]^d)$ can be written as
$$
f(x_1,\ldots,x_d)
= \sum_{q=0}^{2d} \Phi_q\left(\sum_{p=1}^{d}\phi_{q,p}(x_p)\right),
$$
where the outer functions $\Phi_q:\mathbb{R}\to\mathbb{R}$ are continuous and depend on $f$. The important features are: the number of outer summands is finite, the inner functions are universal for the dimension $d$, and the only multivariate operation is addition after applying univariate functions.

This looks superficially similar to a neural network: each term first applies univariate transformations to coordinates, sums them, applies a univariate outer function, and then sums over $q$. Thus the theorem says that composition and summation are, in principle, sufficient to express all continuous multivariate functions. Kůrková used this connection to give a direct neural-network approximation argument with two hidden layers; see [Kůrková 1992](https://doi.org/10.1016/0893-6080(92)90012-8).

However, the theorem is not a practical efficiency theorem for ordinary MLPs. The outer functions $\Phi_q$ depend on the target $f$ and may be highly irregular; the inner functions in the classical theorem are continuous but not the standard activations used in MLPs; and the theorem gives exact representation in terms of arbitrary continuous univariate functions, not parameter counts for ReLU, tanh, or GELU networks. Constructive versions are subtle: [Braun and Griebel 2009](https://doi.org/10.1007/s00365-009-9054-2) discuss constructive proofs and corrections to earlier constructions. Therefore the theorem should not be read as saying that any standard neural network efficiently learns or represents any continuous function.

More recent work revisits this gap between the representation theorem and trainable networks. [Igelnik and Parikh 2003](https://doi.org/10.1109/TNN.2003.813830) proposed Kolmogorov's spline network, using cubic splines to parameterize both inner and outer univariate functions. [Montanelli and Yang 2020](https://doi.org/10.1016/j.neunet.2019.12.013) derive error bounds for deep ReLU networks using a constructive Kolmogorov-Arnold superposition theorem, but only for function subclasses whose outer superposition functions can themselves be efficiently approximated. [Schmidt-Hieber 2021](https://doi.org/10.1016/j.neunet.2021.01.020) emphasizes the same caveat: the classical theorem resembles a two-hidden-layer network syntactically, but the outer functions can be too irregular; modified representations are needed to transfer smoothness to pieces that ReLU networks can approximate. [Polar and Poluektov 2021](https://doi.org/10.1016/j.engappai.2020.104137) study algorithms for constructing Kolmogorov-Arnold-type representations from data. The recent KAN literature takes a more architectural route: [Liu et al. 2024](https://arxiv.org/abs/2404.19756) propose Kolmogorov-Arnold Networks, replacing scalar linear weights and fixed node activations by learnable univariate spline functions on edges; follow-up work applies this idea to scientific computing, for example [Wang et al. 2024](https://doi.org/10.1016/j.cma.2024.117518) on Kolmogorov-Arnold-informed neural networks for PDEs. These papers are inspired by the KA theorem, but their empirical and approximation properties should be evaluated as properties of the proposed spline-edge architectures, not as automatic consequences of the classical theorem.

The useful lesson for MLP expressiveness is conceptual. Kolmogorov-Arnold shows that composition can be universal at the level of continuous functions; modern deep-learning approximation theory asks the sharper quantitative question: when the univariate pieces are restricted to standard activations and finite parameter budgets, which compositional target classes still admit efficient approximation?

### What Section 3 Contributes Beyond Section 2

Section 2 says MLPs are universal; Section 3 asks when they are efficient. The main answers from the literature are:

- smoothness alone gives approximation rates, but generally with dimension dependence;
- Barron or ridge-feature structure can give better high-dimensional rates for two-layer networks;
- compositional or hierarchical structure can make deep networks much more efficient than shallow networks;
- depth separation theorems prove that some functions require exponentially more width when depth is restricted;
- for ReLU-type networks, linear-region counts give a geometric proxy for the extra complexity created by depth.

## 4. More on Expressiveness of MLPs

Section 3 focused on target structure: smoothness, Barron structure, composition, and depth separation. This section separates several other notions that are often mixed together under "expressiveness": minimum width, depth-width tradeoffs, exact piecewise-linear representation, finite-sample memorization, Boolean-domain representation, and combinatorial capacity. These are related, but they answer different questions.

### Width as a Bottleneck

The classical universal approximation theorem lets the hidden-layer width grow. A different question fixes width and lets depth grow. For ReLU networks with scalar output on compact subsets of $\mathbb{R}^d$, the sharp qualitative threshold is width $d+1$: [Hanin and Sellke 2017](https://arxiv.org/abs/1710.11278) show that width $d+1$ is enough for universal approximation, while width at most $d$ is not enough in general. [Lu, Pu, Wang, Hu, and Wang 2017](https://arxiv.org/abs/1709.02540) earlier proved bounded-width universality with width $d+4$ and emphasized the phase transition caused by width.

For vector-valued functions and more general activations, [Kidger and Lyons 2020](https://proceedings.mlr.press/v125/kidger20a.html) prove a deep-narrow universal approximation theorem: for input dimension $n$ and output dimension $m$, width $n+m+2$ is enough for broad continuous nonaffine activations. This result is qualitatively different from the one-hidden-layer theorem: deep narrow networks can be universal even for some activations, such as polynomial activations, that fail the classical shallow non-polynomial condition.

The useful takeaway is that width is not just a proxy for parameter count. Below a critical width, a ReLU network cannot create the needed topology of level sets or decision regions no matter how deep it is. Above that threshold, extra depth can compensate for narrow layers, but the required depth may be very large.

### Depth-Width Tradeoffs

Depth and width can sometimes substitute for each other, but not uniformly. A shallow network can approximate any continuous function if it is allowed to be sufficiently wide. A narrow network can also be universal if it is allowed to be sufficiently deep. But the parameter cost can be very different for specific function families.

Depth separation theorems from Section 3 show one direction: some deep networks require exponentially or super-exponentially many units when represented by shallower networks. [Arora, Basu, Mianjy, and Mukherjee 2018](https://openreview.net/forum?id=B1J_rgWRW) strengthen this line for ReLU networks, giving hard families where reducing depth forces very large increases in size and also studying lower bounds through affine-piece counts. This literature is best read as a resource tradeoff theory: the same target may be cheap in one depth-width regime and expensive in another.

This is why "number of parameters" is not a complete expressiveness measure. Two networks with the same number of parameters can have different depth, bottleneck width, linear-region geometry, and finite-sample interpolation capacity.

### Exact Piecewise-Linear Representation and Sharp Boundaries

For ReLU and leaky-ReLU MLPs, the represented function is continuous piecewise linear. This gives a clean exact-representation target class: continuous piecewise-linear functions. [Arora, Basu, Mianjy, and Mukherjee 2018](https://openreview.net/forum?id=B1J_rgWRW) study this class directly and relate ReLU-network expressiveness to affine pieces and polyhedral geometry.

This viewpoint clarifies step and indicator functions. A ReLU network cannot exactly represent a discontinuous indicator function on a continuous domain, because the network output is continuous. But it can represent or approximate sharp continuous scores whose thresholded classifier has a polyhedral decision boundary. It can also approximate discontinuous targets in $L^p$ or away from the boundary. Thus for classification, one should distinguish the continuous score function from the discontinuous label map obtained after thresholding.

### Finite-Sample Memorization

Finite-sample memorization asks a different question from function approximation: given distinct inputs $x_1,\ldots,x_N$ and arbitrary labels $y_1,\ldots,y_N$, can the network satisfy $f_\theta(x_i)=y_i$ for all $i$? This is interpolation capacity on a finite set, not approximation of a ground-truth function between sample points.

[Zhang, Bengio, Hardt, Recht, and Vinyals 2017](https://research.google/pubs/understanding-deep-learning-requires-rethinking-generalization/) made this distinction central by showing empirically that modern networks can fit random labels, and by giving simple finite-sample expressivity constructions. [Yun, Sra, and Jadbabaie 2019](https://papers.nips.cc/paper/9688-small-relu-networks-are-powerful-memorizers-a-tight-analysis-of-memorization-capacity) give a sharper ReLU memorization theory: three-layer ReLU networks with width on the order of $\sqrt{N}$ can memorize most $N$-point datasets, and they prove matching width-order bounds in that setting. [Vershynin 2020](https://doi.org/10.1137/20M1314884) studies memory capacity for threshold and ReLU networks and connects memorization to the number of connections.

The practical lesson is negative and positive at the same time. Memorization capacity explains why training loss can go to zero even with arbitrary labels. But it does not imply good approximation away from the training set, and therefore it is not a substitute for generalization theory.

### Boolean and Discrete-Domain Functions

Boolean functions on $\{0,1\}^d$ are finite-domain problems. They should not be analyzed in the same way as uniform approximation on $[0,1]^d$. On a finite domain, exact representation is possible by lookup-table or DNF-style constructions, usually with size exponential in $d$ for arbitrary Boolean functions. The interesting question is which Boolean functions have small networks.

This topic connects MLPs to threshold circuits. A threshold gate computes a sign of an affine function, so networks of threshold gates are a discrete analogue of MLPs. Classical references include [Muroga 1971](https://openlibrary.org/books/OL4918111M/Threshold_logic_and_its_applications.) on threshold logic and [Hajnal, Maass, Pudlak, Szegedy, and Turan 1993](https://doi.org/10.1016/0022-0000(93)90001-D) on bounded-depth threshold circuits. A survey-style reference connecting neural networks and Boolean functions is [Anthony 2005](https://researchonline.lse.ac.uk/13924/), with a later book-chapter version in [Anthony 2010](https://www.cambridge.org/core/books/boolean-models-and-methods-in-mathematics-computer-science-and-engineering/neural-networks-and-boolean-functions/ED18747A105CA243BE2BDFBB9C423DC8).

For ReLU networks on Boolean inputs, one can often emulate threshold-like behavior because the input set is finite and margins can be enforced. But lower-bound questions are delicate. [Mukherjee and Basu 2017](https://arxiv.org/abs/1711.03073) study lower bounds over Boolean inputs for networks with ReLU gates, illustrating that discrete-domain expressiveness has its own complexity theory rather than being a direct corollary of continuous universal approximation.

### VC Dimension and Combinatorial Capacity

Another resource measure is the number of labelings a network class can realize on finite samples. This is captured by VC dimension for classifiers and pseudodimension for real-valued function classes. These measures are not the same as approximation error or memorization capacity, but they are useful for comparing architectures as combinatorial classes.

[Bartlett, Harvey, Liaw, and Mehrabian 2019](https://jmlr.org/papers/v20/17-612.html) prove nearly tight VC-dimension and pseudodimension bounds for piecewise-linear networks. If $W$ is the number of weights and $L$ is the number of layers, they give bounds of order $O(WL\log W)$ with matching lower bounds over most parameter regimes. This shows that depth can affect combinatorial capacity even at fixed parameter count, but it still does not by itself explain why overparameterized networks generalize.

### What to Keep Separate

The main point of Section 4 is that several "expressiveness" questions are genuinely different:

- universal approximation asks for density in a function space;
- efficient approximation asks for parameter rates over a target class;
- width-threshold results ask whether depth can compensate for narrow layers;
- depth-separation results compare architectures under constrained resources;
- memorization asks whether arbitrary finite labels can be interpolated;
- Boolean-function representation studies exact computation on $\{0,1\}^d$;
- VC and pseudodimension measure finite-sample combinatorial capacity.

Conflating these leads to misleading statements. For example, a network can be universal but inefficient for a target class; it can memorize finite data but generalize poorly; and it can represent every Boolean function by an exponential-size construction while still failing to represent some structured Boolean functions efficiently at bounded depth.

## 5. Expressiveness of CNNs

CNNs are not just smaller MLPs. They impose a structural hypothesis: nearby coordinates should be processed together, the same local rule should be reused across positions, and the output should often respect a symmetry such as translation equivariance or translation invariance. The basic expressiveness question is therefore not only "are CNNs universal?", but "when does the convolutional bias reduce the number of parameters or samples needed for the relevant target class?"

### Universality of CNNs

CNNs can be universal under suitable architectural assumptions. [Zhou 2020](https://doi.org/10.1016/j.acha.2019.06.004) proves a universal approximation theorem for deep CNNs, showing that convolutional structure does not automatically destroy density in continuous function spaces when depth is allowed to grow. Later work studies more specialized architectures: [Bao, Li, Shen, Tai, Wu, and Xiang 2023](https://doi.org/10.4208/eajam.2022-270.070123) analyze CNNs of the form $g\circ T$, where $T$ is a stack of convolutional layers and $g$ is a fully connected readout, and give conditions under which universality and approximation advantages hold. [Li, Lin, and Shen 2025](https://doi.org/10.1137/23M1570119) study fully convolutional networks for shift-invariant and shift-equivariant functions, proving universal approximation with constant channel width in residual fully convolutional settings and giving necessity results for channel count and kernel size.

The important caveat is that "CNN universality" depends strongly on details: padding, boundary handling, kernel size, channel count, depth, pooling, and whether a final dense layer is allowed. A CNN with a final dense readout can eventually recover arbitrary global dependence, while a fully convolutional architecture without dense readout is naturally constrained toward equivariant or invariant maps. So universality should be stated together with the exact architecture class.

### Locality and Receptive Fields

Locality is the simplest CNN bias. A convolution with a small kernel can only mix nearby coordinates in one layer. After $L$ layers with kernel size $r$, the effective receptive field grows on the order of $L(r-1)$ in one dimension, or analogously in higher-dimensional grids. Thus depth has a concrete spatial role: it lets local information propagate until distant parts of the input can interact.

This helps when the target is local or locally compositional. If the label depends on local patches, a CNN can learn one patch rule and reuse it. If the label depends on features that combine locally into larger features, depth can mirror that hierarchy. But if the target depends on arbitrary long-range interactions with no locality, a small-kernel CNN may need many layers or a dense/global aggregation layer.

### Weight Sharing Versus Local Connectivity

Locality and weight sharing are different biases. A locally connected network uses local receptive fields but has separate weights at each location. A CNN uses local receptive fields and shares the same filter across locations. The distinction matters:

- locality reduces the number of input coordinates each unit sees;
- weight sharing says the same feature detector should apply at many positions;
- pooling or global aggregation can turn equivariant features into invariant outputs.

Recent sample-complexity results separate these effects. [Li, Zhang, and Arora 2021](https://openreview.net/forum?id=uCY5MuAxcxU) give a task where convolutional architectures need far fewer samples than fully connected networks trained by standard orthogonally equivariant algorithms. [Lahoti, Karp, Winston, Singh, and Li 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/71b17f00017da0d73823ccf7fbce2d4f-Abstract-Conference.html) sharpen the comparison using a Dynamic Signal Distribution task: CNNs, locally connected networks, and fully connected networks separate because weight sharing and locality contribute different statistical advantages.

### Equivariance and Invariance

Convolution is naturally translation-equivariant: shifting the input shifts the feature map in the corresponding way. Classification usually wants invariance: shifting an object should not change the class label. Dense prediction tasks such as segmentation usually want equivariance: shifting the image should shift the output mask.

This distinction is a common source of confusion. A stack of convolutions is equivariant, not automatically invariant. Invariance usually requires pooling, global averaging, canonicalization, data augmentation, or a symmetry-constrained readout. Empirically, standard CNN classifiers are not automatically invariant to all translations or small transformations. [Azulay and Weiss 2019](https://jmlr.org/papers/v20/19-519.html) show failures of generalization to small image transformations, and [Biscione and Bowers 2021](https://www.jmlr.org/papers/v22/21-0019.html) show that CNNs are not architecturally invariant to translation in general, though they can learn invariance from suitable data.

The broader equivariant-network literature makes the principle precise. [Cohen and Welling 2016](https://proceedings.mlr.press/v48/cohenc16.html) introduce group-equivariant CNNs, extending weight sharing beyond translations to discrete groups such as rotations and reflections. [Kondor and Trivedi 2018](https://proceedings.mlr.press/v80/kondor18a.html) show, under natural assumptions, that convolutional structure is not only sufficient but necessary for equivariance to compact group actions. [Cohen, Geiger, and Weiler 2019](https://papers.nips.cc/paper_files/paper/2019/hash/b9cfe8b6042cf759dc4c0cccb27a6737-Abstract.html) give a general theory of equivariant CNNs on homogeneous spaces. Approximation-theoretic versions include [Yarotsky 2022](https://doi.org/10.1007/s00365-021-09546-1) for invariant/equivariant maps and [Li, Lin, and Shen 2024](https://www.jmlr.org/papers/v25/22-0982.html) for invariant functions through symmetry-constrained dynamical systems.

### Depth, Pooling, and Tensor Viewpoints

Another literature studies why deep convolutional architectures can be more expressive than shallow ones for compositional data. [Cohen, Sharir, and Shashua 2016](https://proceedings.mlr.press/v49/cohen16.html) connect convolutional arithmetic circuits to hierarchical tensor factorizations: shallow networks resemble CP decompositions, while deep hierarchical networks resemble Hierarchical Tucker decompositions. [Cohen and Shashua 2016](https://proceedings.mlr.press/v48/cohenb16.html) extend this viewpoint to convolutional rectifier networks using generalized tensor decompositions.

Pooling geometry also matters. [Cohen and Shashua 2017](https://openreview.net/forum?id=BkVsEMYel) analyze how pooling structure controls which input partitions can have high separation rank, giving a formal way to say that a convolutional architecture favors some correlation patterns over others. This is a useful bridge between architecture and data: the pooling graph should match the correlation structure of the input domain.

### What Section 5 Contributes

The main CNN expressiveness lessons are:

- CNNs can be universal, but universality depends on exact architectural details;
- locality controls how information flows spatially through depth;
- weight sharing is a separate bias from local connectivity;
- convolution gives equivariance, not invariance by itself;
- pooling and global aggregation decide how local equivariant features become invariant predictions;
- tensor and sample-complexity results show that CNN advantages are strongest when the target has local, repeated, hierarchical, or symmetry-constrained structure.

## 6. More on Expressiveness of CNNs

To understand CNN expressiveness, it is useful to study simple image-inspired target classes rather than arbitrary functions. These toy classes are not meant to model natural images perfectly. They isolate the structures CNNs are designed for: patch locality, repeated local features, local-to-global composition, and translation symmetry.

### Patch and Signal-Detection Functions

A basic model divides the input into $k$ patches, each of dimension $d$:
$$
x=(x^{(1)},\ldots,x^{(k)}),\qquad x^{(i)}\in\mathbb{R}^d.
$$
The target may depend on whether a local signal appears in one of the patches. For example, one can imagine a local template $w_\star$ and a label depending on whether some patch has large correlation with $w_\star$. This captures two image-like facts: the relevant evidence is local, and the same evidence may appear at many positions.

In this setting, a CNN has a natural advantage over a fully connected network because the same filter can be reused across patches. A locally connected network can exploit locality, but if it does not share weights it must learn the same patch rule repeatedly. The Dynamic Signal Distribution task of [Lahoti, Karp, Winston, Singh, and Li 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/71b17f00017da0d73823ccf7fbce2d4f-Abstract-Conference.html) formalizes this separation: for an image with $k$ patches of dimension $d$, CNNs need about $\tilde O(k+d)$ samples, locally connected networks require $\Omega(kd)$ samples in the relevant comparison, and fully connected networks are worse in the locality comparison.

### Local Compositional Functions

Another useful target class consists of functions built by repeatedly combining neighboring features:
$$
f(x_1,\ldots,x_n)
= h_{\mathrm{top}}\bigl(h_1(x_1,x_2), h_2(x_3,x_4), \ldots\bigr),
$$
or more generally by a local tree or grid computation. This is the CNN analogue of the compositional MLP targets discussed in Section 3, but with the additional constraint that the composition respects spatial adjacency.

Deep CNNs match this structure because each layer expands the receptive field and combines nearby information. [Bao, Li, Shen, Tai, Wu, and Xiang 2023](https://doi.org/10.4208/eajam.2022-270.070123) give approximation results showing that CNNs can be more parameter-efficient than fully connected networks for certain compositional target classes. Tensor-decomposition work gives a different formal lens: [Cohen, Sharir, and Shashua 2016](https://proceedings.mlr.press/v49/cohen16.html) and [Cohen and Shashua 2016](https://proceedings.mlr.press/v48/cohenb16.html) connect hierarchical convolutional architectures to tensor factorizations and depth efficiency.

### Translation-Invariant Classification Functions

For classification, a natural target family consists of functions satisfying
$$
f(\tau_s x)=f(x)
$$
for shifts $\tau_s$ in a translation group. This is a model for tasks where moving an object should not change the label. A CNN usually builds translation-equivariant features first and then uses pooling or global aggregation to produce an invariant output.

This distinction matters experimentally and theoretically. Convolutional layers preserve spatial shifts in feature maps, but pooling and boundary effects can break exact equivariance, and invariance is not guaranteed. [Azulay and Weiss 2019](https://jmlr.org/papers/v20/19-519.html) and [Biscione and Bowers 2021](https://www.jmlr.org/papers/v22/21-0019.html) show that standard CNNs can fail to be invariant to translations or small transformations unless the data and training procedure support that invariance. Approximation results for invariant function classes, such as [Yarotsky 2022](https://doi.org/10.1007/s00365-021-09546-1) and [Li, Lin, and Shen 2024](https://www.jmlr.org/papers/v25/22-0982.html), give architecture-level conditions under which invariant/equivariant approximation is complete.

### Shift-Equivariant Dense-Prediction Maps

For segmentation, optical flow, denoising, and many signal-processing tasks, the target is not invariant but equivariant:
$$
F(\tau_s x)=\tau_s F(x).
$$
This says that shifting the input should shift the output. Fully convolutional networks are a natural model class here because they preserve spatial layout at every layer.

[Li, Lin, and Shen 2025](https://doi.org/10.1137/23M1570119) study deep fully convolutional networks from this perspective, proving universal approximation for shift-invariant or shift-equivariant functions in certain residual and non-residual convolutional architectures. This is closer to dense prediction than a CNN followed by a dense classifier, because the architecture itself is constrained to respect the spatial transformation law.

### Pooling Geometry and Correlation Structure

Pooling is not just a computational convenience. It defines how local information is aggregated and which regions of the input are encouraged to interact. [Cohen and Shashua 2017](https://openreview.net/forum?id=BkVsEMYel) formalize this using separation rank: different pooling geometries favor different input partitions, so the architecture's pooling tree encodes an inductive bias about which parts of the image should have strong correlations.

This gives a useful design principle. For ordinary images, local contiguous pooling is natural because nearby pixels and nearby patches tend to form meaningful larger structures. For data with different geometry, such as nonlocal physical variables or graph-like relations, the pooling or aggregation geometry should change accordingly.

### Comparing CNNs, LCNs, and FCNs

A clean expressiveness experiment should compare:

- CNNs: local receptive fields and shared weights;
- locally connected networks: local receptive fields without shared weights;
- fully connected networks: no locality constraint and no sharing.

This comparison separates the effects of locality and weight sharing. [Li, Zhang, and Arora 2021](https://openreview.net/forum?id=uCY5MuAxcxU) show a sample-complexity gap between convolutional and fully connected architectures for a natural distribution under standard training symmetries. [Lahoti, Karp, Winston, Singh, and Li 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/71b17f00017da0d73823ccf7fbce2d4f-Abstract-Conference.html) explicitly separates CNNs, LCNs, and FCNs on an image-like patch task. This is often more informative than comparing CNNs only to MLPs, because it tells us whether the gain comes from locality, sharing, or both.

### Takeaway

The right CNN expressiveness question is not "can CNNs approximate every function?" In many settings they can, after adding enough depth, channels, padding, or dense readout. The sharper question is:

- Does the target have local structure?
- Is the same local rule reused across positions?
- Should the output be invariant or equivariant?
- How large must the receptive field be?
- Does the pooling or aggregation geometry match the target's correlation structure?
- Compared with LCNs and FCNs, which parameter or sample-complexity advantage is actually being tested?

These questions turn CNN expressiveness from a generic universal approximation topic into a theory of architectural bias for spatially structured functions.
