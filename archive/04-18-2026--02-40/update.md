# Update

Plan for writing `## Basic concepts` in [sections/deep_learning.md](/Users/xiaom/Documents/Projects/llm_tutorial/sections/deep_learning.md):

1. Start from the supervised learning setup.
   Introduce the core objects first: input `x`, target `y`, dataset `(x_i, y_i)`, model `f_\theta`, and data distribution. This gives the reader a minimal vocabulary before any deeper discussion.

2. Explain the three basic pieces: task, model, and loss.
   The current subsection titles already suggest this structure, so I would make them the backbone of the section. For each piece, I would give one short definition and one concrete example.

3. Use regression and classification as the first examples.
   These are the simplest ways to explain what prediction means, what ground truth means, and why different tasks need different losses.

4. Introduce optimization only at the intuition level.
   At this stage, I would explain training as "adjust parameters to reduce loss on data" without going deep into gradient descent math yet. The goal is conceptual clarity, not derivation.

5. Preview expressiveness, convergence, and generalization.
   These ideas already appear later in the file, so `## Basic concepts` should only define them informally:
   expressiveness asks what a model can represent, convergence asks whether training can find good parameters, and generalization asks whether performance transfers beyond the training set.

6. Keep notation light and stable.
   I would use one notation consistently throughout the section, avoid introducing too many symbols, and delay more formal statements until later sections.

7. End with a transition to deep learning.
   The section should close by explaining that basic machine learning becomes deep learning when `f_\theta` is a multi-layer compositional function, which connects naturally to the next section.

Target tone:

- concise
- intuitive before formal
- one idea per paragraph
- examples before abstractions
