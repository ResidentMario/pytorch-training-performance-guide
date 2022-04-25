# Home

The **PyTorch Training Performance Guide** is a simple book with a simple objective: documenting commonly used techniques for training [PyTorch](https://pytorch.org/) models to convergence quickly and effectively.

Note that this book is currently in **beta**.

## Why you should read this book

Most deep learning projects start out by training a model to convergence on data specific to your task, then using that model to drive predictions on future data. The more difficult the task, the larger the model needed to perform it, and the longer it takes to train. Many deep learning models used in production systems today have training times measured in days. This has multiple adverse effects:

- High training costs.
- Slow iteration cycles.
- Hardware resource constraints.
- Poor prediction latency once deployed.

These issues are well-documented in production at companies like Google, Facebook, Reddit, etcetera. Luckily they have driven research and development into tools and techniques that can help reduce these costs and remove these barriers.

This book attempts to serve as a simple introduction to the world of "architecture-independent" model optimization from the perspective of a PyTorch practitioner. We cover techniques like model quantization, model pruning, data-distributed training, mixed-precision training, and just-in-time compilation. We include code samples to help you get started and benchmarks showing the power of these techniques on example models of interest.

## Who should read this book

The intended audience for this book are intermediate and advanced deep learning practitioners using PyTorch. This book assumes familiarity with the basics of model training using PyTorch.

Note that the scope of this book is purposefully limited in the following ways:

- We only cover techniques provided in the core PyTorch SDK. While there are many interesting external tools like [Horovod](https://horovod.ai/) and [Apache TVM](https://tvm.apache.org/), the tools built into PyTorch itself are a good place to start.
- We only cover "architecture-independent" ideas. The techniques we cover here should be applicable to almost any model.
- We only cover training, not inference. Model inference optimization is a separate, albeit equally interesting topic. Perhaps a future edition of this book will cover this space also.
- We only cover PyTorch. Many of the same tools and techniques exist in TensorFlow, but (1) I am much less familiar with TensorFlow and (2) it is very difficult to write a technical reference covering two different SDKs at the same time.

All of the code for this book is open source! To submit issues or open pull requests, [visit the GitHub repository](https://github.com/ResidentMario/pytorch-training-performance-guide).

## Table of contents

```{tableofcontents}

```

## Acknowledgements

This book is based on a sequence of blog posts I wrote for the [blog](https://spell.ml/) while working for Spell. We are a model ops company that specializes and making training and deploying deep learning models on the cloud as simple and easy as possible.

I was inspired to collect and edit these disparate posts, plus some new material, into a single book by the blog post [Faster Deep Learning Training with PyTorch – a 2021 Guide](https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/), by Lorenz Kuhn. This blog post was the subject of [one of the most popular r/MachineLearning posts of all time](https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/).
