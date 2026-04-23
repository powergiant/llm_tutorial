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

If we do not know anything about the ground truth, the broadest question is
whether an architecture can approximate every function in a large function
space. For MLPs this usually means approximating every continuous function on a
compact subset of $\mathbb{R}^d$, under a norm such as $L^\infty$.

Universal approximation is a qualitative baseline. It says that the model class
is not fundamentally missing functions in the chosen target space. It does not
say that the approximation is efficient. In high dimension, approximating an
arbitrary function can require exponentially many parameters, so universal
approximation by itself does not solve the practical expressiveness question.

When reading a universal approximation theorem, record:

- the function space being approximated;
- the domain;
- the activation function;
- the norm or topology;
- whether depth, width, or both are allowed to grow.

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
