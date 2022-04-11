# Home

The **PyTorch Training Performance Guide** is a simple book with a simple objective: collecting and explaining commonly used tricks and techniques for training [PyTorch](https://pytorch.org/) models to convergence fast.

## Why training time matters

Most deep learning projects start out by training a model to convergence on data specific to your task, then use that model to drive predictions on that or future data. The more difficult the task, the larger the model that is needed to perform it adequately, and the longer it takes to train it. As a result, many of the deep learning models used in production systems today take a day or more to train just once.

Reducing this training time speeds up the process of trying new things and iterating on your ideas, letting you get better and faster models into production (or onto a research paper) more quickly.

Furthermore, faster training usually means faster imputation and smaller model sizes, allowing you to hit latency and hardware performance targets you might otherwise struggle to hit. Sometimes it's the difference between fitting a model onto GPU and not being able to train it at all.

## Scope and target audience

The intended audience for this book are intermediate and advanced deep learning practitioners using PyTorch. You should already be familiar with the basics of training models using PyTorch.

Techniques covered in this book include model quantization, model pruning, data-distributed training, mixed-precision training, and just-in-time compilation. If any of these ideas sound interesting, this is the book for you!

Note that this book only covers techniques provided in the core PyTorch SDK. There are many other valuable tools in the surrounding ecosystem not covered here, but PyTorch core is a good place to start.

Additionally note that this book will not cover any "architecture-specific" ideas. The techniques we cover here are applicable to almost any model.

## Acknowledgements

This book is based on a sequence of blog posts I wrote for our [blog](https://spell.ml/) while working for Spell. We are a model ops company that specializes and making training and deploying deep learning models on the cloud as simple and easy as possible.

I was inspired to collect and edit these disparate posts into a single book by the blog post [Faster Deep Learning Training with PyTorch – a 2021 Guide](https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/), by Lorenz Kuhn. This led to [one of the most popular r/MachineLearning posts of all time](https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/).

```{tableofcontents}

```
