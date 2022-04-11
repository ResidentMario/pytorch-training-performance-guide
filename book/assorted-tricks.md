# Assorted Tricks

While what constitutes a "trick" certainly varies from person to person; by "trick" here, I mean a simple technique that is straightforward to implement (in a few lines of code) and understand (in a few paragraphs at most). Splitting data loading across multiple processes is a good example, and mixed-precision training is, in my opinion, a good counterexample—though simple to implement, it requires quite a bit of thought to understand. However, again, YMMV.

## Use pinned memory and multiprocessing for data loading

A page is an atomic unit of memory in OS-level virtual memory management. A standard (small, "old-school") page size is `4 KiB`; larger page sizes are possible on modern OSes and systems.

Paging is the act of storing and retrieving memory pages from disk to main memory. Paging is used to allow the working memory set of the applications running on the OS to exceed the total RAM size.

All memory is managed in pages, but paging is only used when the working set spills to disk. **Non-paged memory** is memory that is in RAM, e.g. it has not been spilled to disk.

In CUDA, non-paged CPU (RAM) memory is referred to as **pinned memory**. Pinning a block of memory can be done via a CUDA API call, which issues an OS call that reserves the memory block and sets the constraint that it cannot be spilled to disk.

Pinned memory is used to speed up a CPU to GPU memory copy operation (as executed by e.g. `tensor.cuda()` in PyTorch) by ensuring that none of the memory that is to be copied is on disk. Memory cached to disk has to be read into RAM before it can be transferred to the GPU—e.g. it has to be copied twice. You can naively expect this to be twice as slow (the true slowness depends on the size and business of the relevant memory buses). The following diagram, taken from this blog post, illustrates:

![Pinned memory](/img/ch9/pinned-memory.avif)

The `pin_memory` field (`pin_memory=True`) on `DataLoader` invokes this memory management model. Keep in mind that this technique requires that the OS is willing to give the PyTorch process as much main memory as it needs to complete its load and transform operations—e.g. the batch must fit into RAM in its entirety.

[This SO answer](https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-works-in-dataloader) is also a good reference (note that the `non_blocking` API has been moved; see notes further down).

This is one half of this optimization. The other half is the use of nonblocking data loading.

The default settings for `DataLoader` load the data and executes transforms on it in the model's executing process. This is single-threaded (due to [the GIL](https://www.youtube.com/watch?v=7RlqbHCCVyc)) and blocking.

Setting `num_workers` to a value other than its default 0 will spin up worker processes that individually load and transform the data (e.g. multiprocessing) and load it into the main memory of the host process.

Workers are created whenever an iterator is created from the `DataLoader`, e.g. at enumerate time (presumably by hooking into `__iter__()`). They are destroyed when `StopIterator` is reached, or when the iterator is garbage collected. They receive the following objects from the parent (via IPC): the wrapped `Dataset` object, a `worker_int_fn`, and a `collate_fn`. A `torch.utils.data.get_worker_info` API may be used to write code specific to each worker within these ops.

Data loading via worker processes is trivially easy to do when the root dataset is of a map type—the parent process will handle assigning the indices of the data to load to the workers. Datasets of an iterator type are harder to work with; you will have to use the APIs above to handle slicing them appropriately.

Non-blocking loading has two benefits.

First, it means that any code in between fetching a fresh data batch and executing the `.to('cuda')` call transferring that data to GPU will be concurrent with the loading process, allowing you to do some other work in the main process while dataset processing is still in progress in the worker processes. This would be an ideal place to put e.g. a network call to wandb or something similar.

Second, since batch loading is now split amongst the workers, the process of loading batches of data that are nontrivial in size is made much faster.

The [PyTorch Performance Tuning Guide talk](https://www.youtube.com/watch?v=9mS1fIYj1So) shows the following benchmarks for num_workers optimization of [a trivial MNIST example](https://github.com/pytorch/examples/blob/master/mnist/main.py):

![Num workers optimization](/img/ch9/num-workers.avif)

This script doesn't even use the first optimization, only the second one.

The rule of thumb is to use four times as many processes as you have GPUs.

## Turn on cudNN benchmarking

This optimization is specific to models that make heavy use of convolutional layers (e.g. vanilla convolutional neural networks, or model architecture that feature a CNN backbone).

Convolutional layers use a well-defined mathematical operation, [convolution](https://en.wikipedia.org/wiki/Convolution), which holds foundational importance in a huge variety of applications: image processing, signal processing, statistical modeling, compression, the list goes on and on. As a consequence, a large number of different algorithms have been developed for computing convolutions efficiently on different array sizes and hardware platforms.

PyTorch transitively relies on [NVIDIA's cuDNN framework](https://developer.nvidia.com/cudnn) for the implementations of these algorithms. `cuDNN` has a [benchmarking API](https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936), which runs a short program to chose an algorithm for performing convolution which is optimal for the given array shape and hardware.

You can enable benchmarking by setting `torch.backends.cudnn.benchmark = True`. Thereafter, the first time a convolution of a particular size is run on your GPU device, a fast benchmark program will be run first to determine the best cuDDN convolutional implementation for that given input size is. Thereafter every convolutional operation on a same-sized matrix will use that algorithm instead of the default one.

Again drawing from the [PyTorch Performance Tuning Guide](https://www.youtube.com/watch?v=9mS1fIYj1So) talk, we can see the magnitude of the benefit:

![cuDNN benchmarking](/img/ch9/cudnn-benchmarking.avif)

Keep in mind that using `cudNN` benchmarking will only result in a speed if you keep the input size (the shape of the batch tensor you pass to `model(batch)`) fixed. Otherwise, benchmarking will fire every time input size changes. Since the vast majority of models use a fixed tensor shape and batch size, this shouldn't usually be a problem.

## Use non-blocking device memory transfers

There are a few individual subheadings here.

When creating a new tensor, instead of creating the tensor in host memory and then transferring it to GPU via `.cuda()`, create it directly in CUDA using the `device='cuda'` argument to `torch.tensor`.

When you do transfer memory, it is sometimes useful to enable asynchronous (non-blocking) transfer via `.to(non_blocking=True)`. As long as there is no synchronization point—method call that requires access to this tensor's data, and hence blocks until the transfer is complete—immediately thereafter, this is another way of achieving concurrency.

The most important scenario where this is true is when loading data using `DataLoader` with `pin_memory=True`. Low-level CUDA optimizations [explained here on the NVIDIA blog](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/) allow certain types of data transfers onto the GPU device **from pinned memory only** to execute concurrently with GPU kernel processes. Here is a visualization from that blog post that summarizes how it works:

![Async GPU loading](/img/ch9/async-gpu-loading.avif)

In the first (sequential) example, data loading blocks kernel execution and vice versa. In the latter two (concurrent) examples, the load and execute tasks are first broken down into smaller subtasks, then pipelined in a [just-in-time](https://en.wikipedia.org/wiki/Just-in-time_compilation) manner.

PyTorch uses this feature to pipeline GPU code execution alongside GPU data transfer:

```python
# assuming the loader call uses pinned memory
# e.g. it was DataLoader(..., pin_memory=True)
for data, target in loader:
    # these two calls are also nonblocking
    data = data.to('cuda:0', non_blocking=True)
    target = target.to('cuda:0', non_blocking=True)
    # model(data) is blocking, so it's a synchronization point
    output = model(data)
```

In this example, taken from [this discuss.pytorch.org thread](https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4), `model(data)` is the first synchronization point. Creating the data and target tensors, moving the data tensor to GPU, and moving the target tensor to GPU, and then performing a forward pass in the model, are pipelined for you by PyTorch and CUDA.

The details of how this is done are inscrutable to the end-user. Suffice to say, the end result looks less like Sequential Version and more like Asynchronous Version 2.

Outside of this special case, non-blocking I/O allows you to do some other work in the main process while dataset processing is still in progress in the worker processes. Immediately after some non-blocking data transfer calls would be an ideal place to put e.g. a network call to `wandb` or something similar.

## Try using multiple batches per gradient update

Forward propagating some values through a neural network in train mode creates a computational graph that assigns a weight (a gradient) to every free parameter in the model. Backpropagation then adjusts these gradients to more accurate ("better") values, consuming the computational graph in the process.

However, not every forward pass needs to have a backward pass. In other words, you can call `model(batch)` as many times as you'd like before you finally call `loss.backward()`. The computational graph will continue to accumulate gradients until you finally decide to collapse them all.

For models whose performance is bottlenecked by GPU memory, and hence, batch size—a lot of NLP models, in particular, have this problem—this simple technique offers an easy way to get a "virtual" batch size larger than will fit in memory.

For example, if you can only fit `16` samples per batch in GPU memory, you can forward pass two batches, then backward pass once, for an effective batch size of `32`. Or forward pass four times, backwards pass once, for a batch size of `64`. And so on.

The `huggingface` [Training Neural Nets on Larger Batches](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) blog post provides the following code sample illustrating how this works:

```python
model.zero_grad()                                   # Reset gradients tensors
for i, (inputs, labels) in enumerate(training_set):
    predictions = model(inputs)                     # Forward pass
    loss = loss_function(predictions, labels)       # Compute loss function
    loss = loss / accumulation_steps                # Normalize our loss (if averaged)
    loss.backward()                                 # Backward pass
    if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
        optimizer.step()                            # Now we can do an optimizer step
        model.zero_grad()                           # Reset gradients tensors
        if (i+1) % evaluation_steps == 0:           # Evaluate the model when we...
            evaluate_model()                        # ...have no gradients accumulated
```

Notice that you'll need to combine the per-batch losses somehow. You can usually just take the average.

There's only one downside to using multiple batches instead of one that I'm aware of: any fixed costs that occur during training—latency when transferring data between host and GPU memory, for example—will be paid twice.

## Instead of zeroing out the gradient, set it to None

Proceeding with a new batch of training requires clearing out the gradients accumulated thus far. Historically, this has been done by calling `model.zero_grad()` immediately before calling `model(batch)`. E.g.:

```python
for (X, y) in dataloader:
    # zero out gradients, forward prop...
    model.zero_grad()
    y_pred = model(X)

    # ...then back prop.
    loss = loss_fn(y, y_pred)
    loss.backward()
    optimizer.step()
```

This sets all of the gradients attached to free weights in the model to `0`.

PyTorch 1.7 adds a new option to `zero_grad`: `model.zero_grad(set_to_none=True)`. In this mode gradients are initialized with `None` instead of `0`.

Null initialization is slightly more performant than zero initialization for the following three reasons.

One, zero initialization requires writing a value (`0`) to memory, null initialization does not (presumably it creates a bitwise nullity mask instead).

Two, null initialization will change the first update to the gradient value from a "sum" operation (which requires a read-write) to an "assign" operation (which only requires a write), which is slightly faster.

Three, `torch.optim` optimizers will still perform a gradient update on gradients with value `0`. Though some sophisticated optimizers may still calculate a non-zero weight update in this scenario (due to things like annealing, weighted averaging, weight decay) this is usually not very important to model convergence. `torch.optim` optimizers completely skip updating `None`-valued gradients, saving time.

Due to the third effect, this parameter is technically not side-effect free. Nevertheless, I feel comfortable recommending switching to using m`odel.zero_grad(set_to_none=True)` everywhere by default. You can always try turning it off later, to see if it makes a difference.

## Use gradient clipping

Gradient clipping is the technique, originally developed for handling exploding gradients in RNNs, of clipping gradient values that get to be too large to a more realistic maximum value. Basically, you set a `max_grad`, and PyTorch applies `min(max_grad, actual_grad)` at backpropagation time (note that gradient clipping is bidirectional—a `max_grad` of `10` will ensure that gradient values fall in the range `[-10, 10]`).

The paper [Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity](https://iclr.cc/virtual_2020/poster_BJgnXpVYwS.html) shows (and provides some theoretical justification for why) gradient clipping can improve convergence behavior, potentially allowing you to choose a higher learning rate (and hence converge to an optimized model more quickly):

![Gradient clipping encouraging conversion](/img/ch9/gradient-clipping-convergence.avif)

Gradient clipping in PyTorch is provided via `torch.nn.utils.clip_grad_norm_`. You can apply it to individual parameter groups on a case-by-case basis, but the easiest and most common way to use it is to apply the clip to the model as a whole:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
```

This code sample is taken from the `huggingface` `transformer` codebase [here](https://github.com/huggingface/transformers/blob/7729ef738161a0a182b172fcb7c351f6d2b9c50d/examples/run_squad.py#L156).

What should `max_grad` be set to? Unfortunately, this is not an easy question to answer, as what constitutes a "reasonable" magnitude varies wildly. It's best to treat this value as another hyperparameter to be tuned. The 2013 paper [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) used the value `10` for the intermediate layers and `100` for the output head.

## Disable bias in convolutional layers before batchnorm

Basically every modern neural network architecture uses some form of normalization, with the OG batch norm being the most common.

PyTorch's basic batch norm layer (`torch.nn.BatchNorm2d`) has a `bias` tensor. If the prior layer (1) also has a `bias` tensor (2) applied to the same axis as the batch norm `bias` tensor (3) that is not then squashed by an activation function, the two bias tensors are duplicating work.

In this scenario, it is safe to disable the previous layer's bias term by passing `bias=False` at layer initialization time, shaving some parameters off of total model size.

This optimization is most commonly applicable to convolutional layers, which very often use a `Conv -> BatchNorm -> ReLU` block. For example, [here's a block used by MobileNet](https://github.com/spellml/mobilenet-cifar10/blob/a2914d11f8fdc36618f3d397d9439e9addf5ea16/servers/eval.py#L52):

```python
nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
nn.BatchNorm2d(hidden_dim),
nn.ReLU6(inplace=True),
```

Here's an example of a block where this optimization doesn't apply:

```python
nn.Linear(64, 32),
nn.ReLU(),
nn.BatchNorm2d(),
```

In this case, even though `nn.Linear` does have a bias term on the same axis as `nn.BatchNorm2d`, the effect of that term is being squeezed non-linearly by ReLU, so there is no actual duplicate work.
