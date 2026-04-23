# Expressiveness of MLP and CNN

## Outline

1. Notions of expressiveness
2. Expressiveness of MLPs
3. Depth, width, and representation tradeoffs
4. Step functions, Boolean functions, and finite-sample memorization
5. Expressiveness of CNNs
6. Locality, weight sharing, pooling, and receptive fields
7. Equivariance, invariance, and group-symmetric architectures
8. Metadata to record when surveying papers

## 1. Notions of Expressiveness

Before comparing MLPs and CNNs, it is useful to separate several meanings of
"expressiveness." Different papers may study different notions, so their results
should not be mixed without care.

- Exact representability: whether a network architecture can represent a target
  function exactly.
- Universal approximation: whether a network class is dense in a target function
  space.
- Quantitative approximation: how the approximation error scales with width,
  depth, number of parameters, or smoothness of the target function.
- Separation results: whether one architecture can represent or approximate a
  function much more efficiently than another.
- Finite-sample interpolation: whether a network can fit arbitrary labels on a
  finite dataset.
- Geometric proxies: linear regions, oscillation count, trajectory length, rank,
  and related measures.

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
