# Machine learning

## Basic concepts

[1] Chapter 1-2

* Tasks, ground truth and loss function

    regression, classification, generation


* Model

    $f(x)$ with $(x_i, y_i)$ $y_i = f(x_i)$, $(x_i, s_i)$, $x_i \sim p$

    $|f_\theta(x) - f(x)|^2$, ..., ...

    ...

* Optimization

* Expressiveness, convergence and generalization
    Overfitting
    one parameter one equation intuition
    local minimum intuition

* linear/convex model and their restriction


# General deep learning problem

* model: multi-layer deep nets, compositional function

    * expressiveness: multi-step problem/intrisic simplicity

* optimization

    * Gradient

    * SGD (batch, random batch)

    * AdaGrad

    * Momentum

    * weight regularization

    * Adam/AdamW

    * Auto-differentiation

[1] Chapter 3 + Auto-differentiation


* convergence: 
    shape of loss landscope, SGD avoid some saddle points, but not all (think of optimal for most sample but a small portion). need Momentum. speed need AdaGrad

    gradient vanishing/exposion resnet transformer

* generalization: shape of loss landscope, flat minima, implicit regularization


# Deep learning models

[1] Chapter 4-7

## Every model solves some problem

* Expressiveness of locality -> CNN

* Expressiveness of seq2seq problem -> RNN, LSTM

* Convergence of deep net work, gradient vanishing -> resnet

* gradient vanishing along time direction -> attention

# Basic machine learning theory

TODO: expressiveness, convergence and generalization

TODO: basic proof of generalization, n data n param

TODO: contradiction between them, overparametrization

TODO: case studies of expressiveness, deep learning fits finite step function

TODO: case studies of expressiveness, cnn fits locality function, more efficient

TODO: case studies of expressiveness, transformer, Turing completeness of prompt (in-context learning, difference between linearization), Turing completeness of parameters (meta knowledge, some prompt implies something, then accumulant prompts. even meta meta knowledge, some prompt implies some meta knowledge)

# What next?

pytorch, autograd (why is fast?), tensor calculus

optimization, memory -> infra

# References

[1] [LeeDL Tutorial](https://github.com/datawhalechina/leedl-tutorial)

[2] https://d2l.ai/index.html