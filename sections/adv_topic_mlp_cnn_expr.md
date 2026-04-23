# Expressiveness of MLP and CNN

## Outline

1. Notions of expressiveness
2. Universal approximation
3. Expressiveness of MLPs
4. Expressiveness of CNNs
5. Locality, weight sharing, pooling, and receptive fields
6. Equivariance, invariance, and group-symmetric architectures
7. Metadata to record when surveying papers

TODO: revise this section according to the following

first explain that as discussed in the basic machine learning section, the basic task is we have ground truth $f$, we choose ansatz, ..., data, find parameter from data. question is if the family can really repr or approx the ground truth. 

so we first intro Notions of expressiveness, approximate the ground truth and for small family of ground truth if can cover all in the family using how many parameters 1

if we do not know anything about ground truth, we ask if the family is able to approximate any function this is universal app 2

as in xxx, universal app often requies exponential params, then if we know the ground truth is compositional can we save parameter? also if no prior for any function can mlp improve? 3 (also Depth, width, and representation tradeoffs, and for special examples of functions Step functions, Boolean functions, and finite-sample memorization)

what is locality? what is invariance? if ground truth local and invariance, can cnn save parameters? also analyze expressiveness in some simple cases capture the real picture

## 1. Notions of Expressiveness 




The word "expressiveness" is used in several different ways. For this topic, the
useful notions are the ones that directly compare a network class with a target
function class. A survey should state the notion first; otherwise two papers may
look comparable even though they answer different questions.

### Exact Representability

Exact representability asks whether a target function \(f\) belongs exactly to a
network class \(\mathcal{N}\). In other words, does there exist a choice of
architecture parameters such that \(N(x) = f(x)\) on the whole domain?

This is the strongest notion. It is natural for finite domains, Boolean
functions, piecewise-linear functions, polynomial-like constructions, and
architecture-specific algebraic analyses. It is less natural for discontinuous
functions on continuous domains when the network class is continuous. For
example, a ReLU network cannot exactly represent a discontinuous step function on
all of \(\mathbb{R}^d\), because every ReLU network is continuous.

When reading an exact-representation result, record the domain carefully. Exact
representation on \(\{0,1\}^d\) is very different from exact representation on a
continuous subset of \(\mathbb{R}^d\).

### Universal Approximation

Universal approximation asks whether a network class can approximate every
function in a target function space arbitrarily well. Formally, for each target
function \(f\), error tolerance \(\epsilon > 0\), and norm or metric \(d\), does
there exist a network \(N\) such that \(d(N, f) < \epsilon\)?

This notion answers a qualitative question: is the architecture expressive
enough in principle? It does not say whether the required network is small,
trainable, stable, or practical. A one-hidden-layer MLP may be universal, but the
width required for a given accuracy can be enormous.

Important details:

- Target space: continuous functions, \(L^p\) functions, measurable functions,
  invariant functions, equivariant functions, or another class
- Domain: compact subset of \(\mathbb{R}^d\), discrete cube, image grid, or group
  domain
- Error notion: uniform norm, \(L^p\) norm, weak approximation, or finite-sample
  error
- Architecture constraints: fixed depth, bounded width, convolutional structure,
  equivariance, pooling, or channel restrictions

### Quantitative Approximation

Quantitative approximation asks how efficiently a network approximates a target
function class. Instead of only asking whether approximation is possible, it asks
how the error decreases as the network size increases.

A typical result has the form: functions in some class \(\mathcal{F}\) can be
approximated to error \(\epsilon\) by networks with depth \(L\), width \(W\), or
parameter count \(P\), where these quantities scale in a specified way with
\(\epsilon\), dimension \(d\), and smoothness or structural assumptions on
\(\mathcal{F}\).

This is often more informative than universal approximation because it exposes
the cost of approximation. It can show, for example, whether an architecture
suffers from the curse of dimensionality, whether compositional structure helps,
or whether convolutional locality gives a parameter advantage for spatially
structured functions.

When comparing quantitative results, do not compare only the final rate. Also
record the target function class, norm, allowed weight sizes, depth, width,
number of parameters, and whether the construction is explicit.

### Efficient Representation and Separation

Separation results compare two architectures or two resource regimes. The goal is
to show that one network class represents or approximates some functions much
more efficiently than another.

Typical examples:

- Deep networks versus shallow networks
- Narrow deep networks versus wide shallow networks
- CNNs versus fully connected networks
- CNNs versus locally connected networks without weight sharing
- Overlapping convolutional architectures versus non-overlapping ones

A separation result usually has two parts: an upper bound for the stronger
architecture and a lower bound for the weaker architecture. For example, a deep
network may approximate a function with polynomially many parameters, while any
shallow network needs exponentially many parameters.

These results are useful because they formalize statements such as "depth helps"
or "locality helps." However, the hard functions used in separations may be
special constructions, so the survey should distinguish worst-case separation
from evidence about typical practical tasks.

### Finite-Sample Interpolation and Memorization

Finite-sample interpolation asks whether a network can fit arbitrary labels on a
finite set of inputs. Given samples \((x_i, y_i)_{i=1}^n\), does there exist a
network \(N\) such that \(N(x_i) = y_i\) for every sample?

This notion is useful for understanding overparameterization and memorization.
It is not the same as universal approximation. Interpolating finitely many
points says that the network can fit a dataset, but it says little by itself
about behavior away from those points or about generalization.

Important assumptions include:

- Whether the inputs are distinct, separated, or in general position
- Whether labels are scalar, vector-valued, Boolean, or real-valued
- How many parameters or neurons are needed as a function of sample size
- Whether the interpolation construction is robust to perturbations

### Auxiliary Proxies

Some papers use proxies such as number of linear regions, trajectory length,
oscillation count, tensor rank, or separation rank. These quantities are useful
diagnostics, but they should not be treated as expressiveness itself unless they
are connected to an approximation or representation theorem.

For example, a larger number of linear regions suggests greater geometric
complexity, but it does not automatically imply better approximation of a target
function class. In this note, such quantities should be recorded as supporting
tools rather than primary notions.

## 2. Expressiveness of MLPs

### Classical Universality

The first MLP topic is the classical universal approximation theorem. Important
questions include:

- Which activation functions are sufficient for universality?
- Is the target function continuous, measurable, or in an \(L^p\) space?
- Is the domain compact or non-compact?
- Does the theorem require one hidden layer, arbitrary width, or arbitrary depth?

This section should include Cybenko, Hornik, Leshno-type results, and later
refinements.

### Minimal Conditions for Universality

Beyond the existence of universal approximation, a sharper question is what
architectural conditions are necessary.

- Minimal width needed for ReLU networks to be universal
- How depth can compensate for limited width
- What fails when width is below the critical threshold
- Whether bounded-width networks can still be universal with sufficient depth

### Quantitative Approximation Theory

Universality alone does not explain efficiency. A more informative theory asks
how many neurons or parameters are needed to approximate structured function
classes.

Important target classes include:

- Barron or Fourier-type function classes
- Sobolev, Besov, and Holder classes
- Analytic functions
- Functions supported near low-dimensional manifolds
- Compositional or hierarchical functions

## 3. Depth, Width, and Representation Tradeoffs

### Depth-Separation Results

This is the main formal version of the claim that multiple layers improve
expressiveness. The goal is to identify functions that a deep network represents
efficiently but any shallow network represents only with exponentially many
units or parameters.

Representative themes:

- Eldan-Shamir type separation between two-layer and three-layer networks
- Telgarsky oscillation and sawtooth constructions
- Safran-Shamir style separations for natural geometric functions
- Radial, ball-indicator, or compositional hard examples

### Compositional Structure

Depth is especially useful when the target function has a hierarchical or
compositional structure. A deep network can mirror the structure of the target,
while a shallow network may need many more units.

Survey questions:

- What compositional assumptions are made?
- Are the approximation rates dimension-dependent or dimension-free?
- Is the result constructive?
- Does the theorem distinguish depth from parameter count?

### Complexity Measures

Different papers measure network size differently. A useful survey should record
which measure is used.

- Depth
- Width
- Number of neurons
- Number of nonzero parameters
- Weight magnitude or bit complexity
- Number of linear regions
- Rank or separation rank

## 4. Step Functions, Boolean Functions, and Memorization

The original topic "MLP fits finite step function, finite step Boolean function"
should be split into several separate questions.

### Step, Indicator, and Piecewise-Constant Functions

ReLU networks are continuous functions on \(\mathbb{R}^d\), so discontinuous
step functions are usually approximation targets rather than exact
representation targets on continuous domains.

Topics to separate:

- Indicators of intervals, halfspaces, balls, and polytopes
- Piecewise-constant functions on continuous domains
- Approximation in \(L^p\), uniform, or weak senses
- Exact representation after restricting the domain or using discontinuous
  activations

### Boolean Functions

Boolean functions on \(\{0,1\}^d\) are different from continuous-domain step
functions. Since the domain is finite, exact representation becomes possible in
ways that do not contradict continuity on \(\mathbb{R}^d\).

Important connections:

- Threshold circuits
- Parity, majority, disjointness, and inner-product functions
- Exact representation of arbitrary Boolean functions
- Lower bounds for shallow threshold, ReLU, or polynomial threshold circuits

### Finite-Sample Interpolation and Memorization

This line studies the ability of a network to fit arbitrary labels on a finite
set of samples.

Questions to track:

- How many samples can a network memorize?
- How many neurons, layers, or parameters are needed?
- Does depth reduce the interpolation cost?
- Are the inputs assumed to be in general position?
- Is the construction stable or merely existential?

## 5. Expressiveness of CNNs

### CNN Universality

The first CNN question is whether convolutional architectures are universal
approximators under suitable assumptions.

Important variables:

- Kernel size
- Number of channels
- Depth
- Padding and boundary treatment
- Downsampling or pooling
- Whether the final layers are convolutional or fully connected

### Approximation Advantages on Structured Targets

CNNs are most compelling when the target function has spatial or local
structure. This section should focus on when convolutional networks approximate
such functions more efficiently than generic fully connected networks.

Examples of structure:

- Local interactions
- Translation-related patterns
- Sparse or hierarchical spatial dependencies
- Radial or compositional functions
- Multiscale structure

### CNN Linear-Region Complexity

For ReLU CNNs, one can study expressiveness through the number of linear
regions, similar to the MLP literature.

Key questions:

- How many linear regions can a CNN create?
- How does this compare with a fully connected ReLU network using the same
  number of parameters?
- How do convolution, channel count, and depth affect the count?

## 6. Locality, Weight Sharing, Pooling, and Receptive Fields

CNN expressiveness should not be treated as one single phenomenon. Several
architectural ingredients should be separated.

### Locality

Locality means each unit only sees a local neighborhood of the previous layer.
It can reduce the number of parameters and match local structure, but may also
restrict information flow unless depth is sufficient.

Survey questions:

- When does locality help approximation efficiency?
- When does locality restrict representability?
- How much depth is needed for global interactions?

### Weight Sharing

Weight sharing distinguishes CNNs from locally connected networks. It imposes a
translation-related structural constraint and can improve parameter efficiency.

Compare:

- Fully connected networks
- Locally connected networks without weight sharing
- Convolutional networks with shared weights

### Overlapping Receptive Fields

Overlapping receptive fields are not merely an implementation detail. They can
change the expressive power of the architecture.

Questions:

- Do overlapping filters give exponential gains over non-overlapping filters?
- How does overlap affect depth efficiency?
- How does it interact with pooling?

### Pooling

Pooling choices can change what functions a CNN can represent efficiently.

Topics:

- Max pooling versus average pooling
- Pooling as a source of invariance
- Pooling in tensor-decomposition analyses
- Whether pooling loses information needed for exact representation

### Tensor and Rank Viewpoints

Some CNN expressiveness results are best understood through tensor
decompositions and rank measures.

Important notions:

- Convolutional arithmetic circuits
- Separation rank
- Hierarchical tensor decompositions
- Complete and incomplete depth efficiency

## 7. Equivariance, Invariance, and Group-Symmetric Architectures

### Shift-Equivariant CNNs

Fully convolutional networks naturally represent shift-equivariant maps. This
should be separated from generic CNN universality.

Questions:

- Which shift-equivariant functions can fully convolutional networks
  approximate?
- What channel and kernel-size conditions are necessary?
- When is a fully convolutional architecture universal within the class of
  equivariant functions?

### Group-Equivariant and Invariant Networks

The broader equivariant-network literature extends CNN ideas from translations
to general groups.

Topics:

- Universality for invariant networks
- Universality for equivariant networks
- When higher-order tensor features are required
- Distinction between separating group orbits and approximating all continuous
  invariant functions

## 8. Metadata for Paper Survey

For every paper in the eventual survey, record the following information.

- Architecture: MLP, CNN, locally connected network, fully convolutional
  network, group-equivariant network, or another variant
- Activation function: ReLU, sigmoid, threshold, polynomial, maxout, or other
- Domain: \(\mathbb{R}^d\), compact subset of \(\mathbb{R}^d\),
  \(\{0,1\}^d\), grid/image domain, or group domain
- Target function class: continuous, \(L^p\), Sobolev, Boolean,
  compositional, invariant, equivariant, finite sample labels
- Expressiveness notion: exact representation, universal approximation,
  quantitative rate, interpolation, linear regions, rank, or separation theorem
- Error metric: uniform norm, \(L^p\), classification error, weak
  approximation, finite-sample loss, or another metric
- Complexity measure: depth, width, neurons, parameters, nonzero parameters,
  channels, kernel size, rank, or weight magnitude
- Result type: upper bound, lower bound, equivalence, separation, or
  construction
- Main limitation: restrictive assumptions, nonconstructive proof, large
  constants, special target function, or mismatch with practical CNNs

## Condensed Survey Map

- MLP universality
- Minimal width and depth conditions for MLP universality
- Quantitative MLP approximation rates
- Depth-separation theorems
- Compositional and hierarchical target functions
- Exact versus approximate representation of step and indicator functions
- Boolean functions and circuit-complexity connections
- Finite-sample interpolation and memorization
- Linear-region and geometric expressiveness proxies
- CNN universality
- CNN approximation advantages on structured targets
- Locality, weight sharing, and comparisons with fully connected or locally
  connected networks
- Overlapping receptive fields
- Pooling effects
- Tensor and rank formulations of CNN expressiveness
- CNN linear-region complexity
- Shift-equivariant fully convolutional universality
- Group-equivariant and invariant universality
