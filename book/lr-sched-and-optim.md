# LR Schedulers, Adaptive Optimizers

A long long time ago, almost all neural networks were trained using a fixed learning rate and the stochastic gradient descent (SGD) optimizer.

Then the whole deep learning revolution thing happened, leading to a whirlwind of new techniques and ideas. In the area of model optimization, the two most influential of these new ideas have been **learning rate schedulers** and **adaptive optimizers**.

In this chapter, we will discuss the history of learning rate schedulers and optimizers, leading up to the two techniques best-known among practitioners today: `OneCycleLR` and the `Adam` optimizer. We will discuss the relative merits of these two techniques.

**TLDR**: you can stick to `Adam` (or one of its derivatives) for the most part, but you should try `OneCycleLR` eventually.

## ReduceLROnPlateau, the first LR scheduler

All optimizers have a **learning rate** hyperparameter, which is one of the most important hyperparameters affecting model performance.

In the simplest case, the learning rate is kept fixed. However, it was discovered relatively early on that choosing a large initial learning rate then shrinking it over time leads to better converged, more performant models. This is known as **learning rate annealing** (or decay).

In the early stages of model training, the model is still making large steps towards the gradient space, and a large learning rate helps it find the coarse values it needs more quickly.

In the late stages of model training, the opposite is true. The model has approximately the right gradients already; it just needs a little extra push to find the last few percentage points of performance. A large gradient is no longer appropriate because it will "overshoot" the point of optimality. Instead of converging on the global cost minima, the model will bounce around it:

![An unstable learner that is missing the true local minima.](img/ch1/unstable-learner.avif)

This observation led to the popularization of the first well-known learning rate scheduler, `ReduceLROnPlateau` (`torch.optim.lr_scheduler.ReduceLROnPlateau` in PyTorch). `ReduceLROnPlateau` takes a `step_size`, a `patience`, and a `cooldown` as input. After completing each batch of training, the model checks whether or not model performance has improved. If model performance hasn't improved in `patience` batches, the learning rate is reduced (typically by a factor of 10). After a cooldown period, this process is repeated again, until the final batch of training completes.

This technique has squeezed out an extra percentage point or two of performance in pretty much every context it's been tried in. As a result, some combination of `EarlyStopping`, `ReduceLROnPlateau`, and `SGD` was the state of the art until 2015.

## Adaptive optimizers

2015 saw the release of Adam: A Method For Stochastic Optimization. This paper introduced `Adam` (`torch.optim.Adam` in PyTorch), the first so-called adaptive optimizer to gain widespread traction.

**Adaptive optimizers** eschew the use of a separate learning rate scheduler, instead embedding learning rate optimization directly into the optimizer itself. `Adam` actually goes one step further, managing the learning rates on a per-weight basis. In other words, it gives every free variable in the model its own learning rate. The value `Adam` actually assigns to this learning rate is an implementation detail of the optimizer itself, and not something you can manipulate directly.

`Adam` has two compelling advantages over `ReduceLROnPlateau`.

One, model performance. It's a better optimizer, full stop. Simply put, it trains higher-performance models.

Two, `Adam` is almost parameter-free. `Adam` does have a learning rate hyperparameter, but the adaptive nature of the algorithm makes it quite robust—unless the default learning rate is off by an order of magnitude, changing it doesn't affect performance much.

`Adam` is not the first adaptive optimizer—that honor goes to `adagrad`, published in 2011—but it was the first one robust enough and fast enough for general-purpose usage. Upon its release, `Adam` immediately overtook `SGD` plus `ReduceLROnPlateau` as the state of the art in most applications. We've seen improved variants (like `Adamw`) since then, but these have yet to displace vanilla Adam in general-purpose usage.

Because `Adam` manages learning rates internally, it's incompatible with most learning rate schedulers. Anything more complicated than simple learning warmup and/or decay will cause the `adam` optimizer to "complete" with the learning rate scheduler when managing its internal LR, causing model convergence to worsen.

This makes `Adam` fundamentally incompatible with the two techniques we'll cover in the next two sections of this post, cosine annealing and one-cycle learning, both learning rate schedulers that do tricky and interesting things to your learning rate.

## Cosine annealed warm restart

The 2017 paper [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) popularized the idea of warm restarts. A learning rate scheduler that incorporates warm restarts occasionally re-raises the learning rate. A simple linear example showing how this is done:

![An annealled sawtooth learning rate.](img/ch1/lr-sawtooth.avif)

Warm restarts usually actually cause the model to diverge. This is done on purpose. It turns out that adding some controlled divergence allows the model to work around local minima in the task's cost surface, allowing it to find an even better global minima instead. This is akin to finding a valley, then climbing a nearby hill, and discovering an even deeper valley one region over. Here's a visual summary:

![Two learners moves around a cost surface.](img/ch1/cost-surface-discovery.avif)

Both of these learners converge to the same global minima. However, on the left, the learner trundles slowly along a low-gradient path. On the right, the learner falls into a sequence of local minima (valleys), then uses warm restarts to climb over them (hills). In the process it finds the same global minima faster, because the path it follows has a much higher gradient overall.

[fast.ai](https://www.fast.ai/) popularized a learning rate scheduler that uses both warm restarts and cosine annealing. This scheduler has the following shape:

![Two learners moves around a cost surface.](img/ch1/cosine-annealing.avif)

Cosine annealing has better convergence behavior than linear annealing, for reasons that are not entirely understood.

This learning rate scheduler was the default one used by the `fastai` framework for a couple of years. It was first made available in PyTorch (as `torch.optim.lr_scheduler`.CosineAnnealingLR) in version 0.3.1, released in February 2018 ([release notes](https://github.com/pytorch/pytorch/releases/tag/v0.3.1), [GH PR](https://github.com/pytorch/pytorch/pull/3311)).

## One-cycle learning rate schedulers

`fastai` no longer recommends cosine annealing because it is no longer the most performant general-purpose learning rate scheduler. These days, that honor belongs to the one-cycle learning rate scheduler.

The one-cycle learning rate scheduler was introduced in the 2017 paper [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120). The paper uses the following learning rate policy (which is applied over both learning rate and momentum):

![A figure from the one-cycle paper illustrating the one-cycle LR scheduler pattern.](img/ch1/one-cycle-paper-figure.avif)

Optimally, learning rate and momentum should be set to a value which just causes the network to begin to diverge at its peak. The remainder of the training regimen consists of warm-up, cool-down, and fine-tuning periods. Note that, during the fine-tuning period, the learning rate drops to 1/10th of its initial value.

Momentum is counterproductive when the learning rate is very high, which is why momentum is annealed in the opposite of the way in which the learning rate is annealed in the optimizer.

The one-cycle learning rate scheduler uses more or less the same mechanism that the cosine annealed warm restarts learning rate scheduler uses, just in a different form factor.

In their implementation, `fastai` tweaks this a little bit, again switching from linear to cosine annealing:

![A figure from the one-cycle paper illustrating the one-cycle LR scheduler pattern.](img/ch1/fastai-one-cycle-behavior.avif)

`fastai` recommends the one-cycle LR scheduler plus `SGD` over `Adam` because, subject to some tuning (getting the maximum learning rate correct is particularly important), it trains models with roughly equal or marginally worse performance in a fraction of the time. This is due to a phenomenon that Leslie Smith, the once-cycle paper author, refers to as superconvergence. For example, the paper shows the following behavior on CIFAR10:

![A figure from the one-cycle paper illustrating the one-cycle LR scheduler pattern.](img/ch1/one-cycle-cifar10-perf.avif)

Though you shouldn't count on results this compelling in practice, superconvergence has indeed been demonstrated on a broad range of datasets and problem domains.

The one-cycle learning rate scheduler was implemented in PyTorch in August 2019 (as `torch.optim.lr_scheduler.OneCycleLR`; here's the [GH PR](https://github.com/pytorch/pytorch/pull/21258)).

## The view from 2021

Despite `fastai` advocacy, at present, most practitioners still use the `Adam` optimizer as their default.

For example, if you browse recent starter kernels for Kaggle competitions, [like this one](https://www.kaggle.com/gogo827jz/jane-street-neural-network-starter), you'll see that the use of `Adam` predominates. As I explained above, in large part because `Adam` is pretty much parameter-free, it's much more robust to model changes than `OneCycleLR` is. This makes it much easier to develop with, as it's one fewer set of hyperparameters that you have to optimize. These benefits seem to have proven to be more important than the computational benefit of `OneCycleLR`, hyperconvergence, in practice.

However, once you're in the later optimization stage of a medium-sized model training project, experimenting with moving off of `Adam` and onto `OneCycleLR` is well worth doing. Just imagine how much easier the lives of your data engineers will be if your model can achieve 98% of the performance in 25% of the time!

Hyperconvergence is an extremely attractive property to have, if you can spend the time required to tune it.

## To-do

- Query practitioners for more detailed information on the trade-off b/c the conclusion (and recommendation) is currently very weak.
- Peformance benchmarks.
