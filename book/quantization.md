# Quantization

Quantization is a fairly recent technique for speeding up deep learning model inference time. This technique has become very popular very quickly because it has been shown to provide impressive improvements in model performance in both research and production settings. For example, in their article [How We Scaled Bert To Serve 1+ Billion Daily Requests on CPUs](https://medium.com/roblox-tech-blog/how-we-scaled-bert-to-serve-1-billion-daily-requests-on-cpus-d99be090db26), the Roblox engineering team discusses how they were able to leverage quantization to improve their throughput by a factor of 10x:

![Roblox model serving time improvements from their blog post](/img/ch5/roblox-serving-improvements.avif)

How does it work? Well, feeding an input to a deep learning model and getting a result back out boils down to a long sequence of vector math operations. Quantization works by simplifying the data type these operations use. In PyTorch, this means converting from default 32-bit floating point math (`fp32`) to 8-bit integer (`int8`) math. `int8` has a quarter as many bits as `fp32` has, so model inference performed in `int8` is (naively) four times as fast.

This chapter in an introduction to the quantization techniques available in PyTorch. We will:

- Discuss the motivation for using quantization
- Introduce (and demonstrate) the three forms of quantization built into PyTorch
- Run some benchmarks to see how it performs.

All of the model code is available on GitHub: [here](https://github.com/spellml/tweet-sentiment-extraction/blob/master/servers/eval_quantized.py), [here](https://github.com/spellml/unet-bob-ross/blob/master/servers/eval_quantized.py), and [here](https://github.com/ResidentMario/mobilenet-cifar10/blob/master/servers/eval_quantized.py).

## How quantization works

Before we can understand how mixed precision training works, we first need to review a little bit about numerical types.

In computer engineering, decimal numbers like `1.0151` or `566132.8` are traditionally represented as floating point numbers. Since we can have infinitely precise numbers (think `π`), but limited space in which to store them, we have to make a compromise between precision (the number of decimals we can include in a number before we have to start rounding it) and size (how many bits we use to store the number).

The technical standard for floating point numbers, IEEE 754 (for a deep dive I recommend the PyCon 2019 talk [Floats are Friends: making the most of IEEE754.00000000000000002](https://www.youtube.com/watch?v=zguLmgYWhM0)), sets the following standards:

- `fp64`, aka double-precision or "double". `float` in Python uses this type.
- `fp32`, aka single-precision or "single". PyTorch uses this type by default.
- `fp16`, aka half-precision or "half".

Floating points need a specification because operating on and storing unbounded numbers is complicated. Integer numbers like `1`, `-12`, or `42`, are comparatively simple. An `int32`, for example, has 1 bit reserved for the sign, and 31 bits for the digits. That means it can store `2^31 = 4294967296` total values, ranging from `-2^31 to 2^31 - 1`. The same logic holds for an `int8`: this type holds `2^8 = 256` total values in the range `-2^7 = -128` through `2^7 - 1 = 127`.

Quantization works by mapping the (many) values possible in `fp32` onto the (just `256`) values possible in `int8`. This is done by binning the values: mapping ranges of values in the `fp32` space into individual int8 values. For example, two weights constants `1.2251` and `1.6125` in `fp32` might both be converted to `12` in `int8`, because they are both in the bin `[1, 2]`. Picking the right bins is obviously very important.

**PyTorch provides three different quantization algorithms, which differ primarily in where they determine these bins**—"dynamic" quantization does so at runtime, "training-aware" quantization does so at train time, and "static" quantization does so as an additional intermediate step in between the two. Each of these approaches has advantages and disadvantages (which we will cover shortly). Note that there are other quantization techniques proposed in the academic literature as well.

Once the values (weights, inputs, and intermediate vectors) have been converted into `int8` format, most of the math that follows is performed in `int8` (an exception is made for certain accumulation operations, e.g. `sum`, which accumulate error especially quickly). This type has 25% as many bits as the default type, resulting in the following desirable properties:

- Reduction in model size that asymptotically approaches 4x
- 2-4x reduction in memory bandwidth
- 2-4x faster inference due to savings in memory bandwidth and compute

## Quantization in practice

There are a number of caveats to this improved performance in practice.

**Quantization is an inference-only technique**. `int8` is not numerically accurate enough to support backpropagation. Such aggressive rounding—from fine-grained floating point values to integer approximations—introduces inaccuracy into the model. Training is much more sensitive to weight inaccuracy than serving; performing backpropagation in `int8` will almost assuredly cause the model to diverge. A similar but less invasive technique, mixed-precision training, is used instead.

**Not all models are equally sensitive to quantization**. Quantization is fundamentally an approximation technique, and hence always reduces model performance, but the extent of the regression is highly model-dependent. Performance regression in practice can range anywhere from >10% to 0.1%, depending on model robustness, the choice of technique, and how much of the model you quantize. Here, "robustness" is usually analogous to "model size": a large model with many redundant connections will typically perform better than a smaller one with just a few sparse connections.

**Quantization need not be applied to the entire model**. It is possible to run certain parts of the network in `int8`, but leave other parts in `fp32`. A relatively cheap conversion operation is inserted between the `int8` and `fp32` segments. This can be used to tune the performance of models that don't respond well to one-shot quantization.

**Not all layers can be quantized**. Some layers accumulate error too quickly when quantized to be useful (e.g. accumulation operations). Others simply haven't been implemented yet because the API is so new. There is no master list (that I'm aware of) of which operations have quantized implementations and which ones don't, so discovering this is currently mostly a matter of trial and error, unfortunately.

**Quantization in PyTorch is currently CPU-only**. Quantization is not a CPU-specific technique (e.g. NVIDIA's TensorRT can be used to implement quantization on GPU). However, inference time on GPU is already usually "fast enough", and CPUs are more attractive for large-scale model server deployment (due to complex cost factors that are out of the scope of this article). Consequently, as of PyTorch 1.6, only CPU backends are available in the native API.

In the sections that follow, we will introduce and review the techniques one at a time.

## Dynamic quantization

Dynamic quantization is the easiest form of quantization to use. In fact it is so easy to use that here is the entire API expressed in a single code sample:

```python
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

In this code sample:

- `model` is the PyTorch module targeted by the optimization.
- `{torch.nn.Linear}` is the set of layer classes within the model we want to quantize.
- `dtype` is the quantized tensor type that will be used (you will want `qint8`).

What makes dynamic quantization "dynamic" is the fact that it fine-tunes the quantization algorithm it uses at runtime. Recall that quantizing a fine-grained `fp32` vector requires choosing a set of `int8` bins and an algorithm for splitting those into those bins. Dynamic quantization simply multiplies input values by a scaling factor, then rounds the result to the nearest whole number and stores that.

Model weights (which are known fixed ahead of time) are quantized immediately; activations are quantized using this dynamic algorithm at runtime, with small adjustments to the scaling factor made based on the input values observed, until the conversion operation is approximately optimal.

This very simple on-the-fly approach doesn't require making very many choices, which is what allows PyTorch to provide it in the form of a one-shot function wrapper API.

Dynamic quantization is the least performant quantization technique in practice—e.g., it is the one that will have the most negative impact on your model performance. This is made up for by its simplicity: you can kind of chuck it at your model and see if it works. In practice, dynamic quantization performance is still more than adequate for large (typically server-side) NLP models where the memory bandwidth of the weights is the performance bottleneck: LSTMs, RNNs, and Transformer architectures.

## Static quantization

Static quantization (also called post-training quantization) is the next quantization technique we'll cover.

Static quantization works by fine-tuning the quantization algorithm on a test dataset after initial model training is complete. This additional scoring process is not used to fine-tune the model—only to adjust the quantization algorithm parameters. This is much more involved than dynamic quantization, requiring an additional pass over the dataset to work, but it's much more accurate: static quantization gives the algorithm the opportunity to calibrate using real data all at once, instead of having to do so one-at-a-time at run time.

Static quantization requires changes to your model code. The module initialization code needs `torch.quantization.QuantStub` and `torch.quantization.DeQuantStub` layers inserted into the model. For example:

```python
def __init__(self):
        super(M, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()
```

In this example, the quant layer will perform the `fp32 -> int8` conversion, `conv` and `relu` will run in `int8`, and then the dequant layer will `int8 -> fp32` convert the input value back for emission.

In dynamic quantization, only layers belonging to the set of types we pass to the function are quantized—the API is opt-in. Static quantization, by contrast, automatically applies quantization to all layers that can be quantized. To opt out of quantization for a specific layer, you need to set its `qconfig` field to `None—e`.g. `model.conv_1_4.qconfig = None`. You will need to insert `QuantStub` and `DeQuantStub` layers yourself as needed, to control the model's quantization boundaries to match.

Another thing you have to be aware of when using static quantization is the backend. PyTorch uses one of two purpose-built reduced-precision tensor matrix math libraries: FBGEMM on `x86` ([repo](https://github.com/pytorch/FBGEMM)), QNNPACK ([repo](https://github.com/pytorch/pytorch/tree/169541871a7a6663cc86c3ab68501a62a5d8c67c/aten/src/ATen/native/quantized/cpu/qnnpack)) on ARM. These are designed to use the PyTorch tensor format, e.g. they do not need to convert input tensors to their own internal representation (slowing down processing).

Since these libraries are architecture-dependent, **static quantization must be performed on a machine with the same architecture as your deployment target**. If you are using FBGEMM, you must perform the calibration pass on an `x86` CPU (usually not a problem); if you are using QNNPACK, calibration needs to happen on an `ARM` CPU (this is harder).

Finally, to get the most performance out of static quantization, you need to also use module fusion. **Module fusion** is the technique of combining ("fusing") sequences of high-level layers, e.g. `Conv2d` + `Batchnorm`, into a single combined layer. This improves performance by pushing the combined sequence of operations into the low-level library, allowing it to be computed in one shot, e.g. without having to surface an intermediate representation back to the PyTorch Python process. This speeds things up and leads to more accurate results, albeit at the cost of debuggability.

Module fusion is performed using `torch.quantization.fuse_modules`, which takes named module layers as input:

```python
model = torch.quantization.fuse_modules(model, [['conv', 'relu']])
```

At the time of writing, module fusions is only supported for a handful of very common CNN layer combinations: `[Conv, Relu]`, `[Conv, BatchNorm]`, `[Conv, BatchNorm, Relu]`, `[Linear, Relu]`. There are also some differences between which combinations of layers the two different backends support, so YMMV.

Here's a code sample, taken from the PyTorch docs, showing the full static quantization process:

```python
model_fp32 = M()
model_fp32.eval()
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_fp32_fused = torch.quantization.fuse_modules(
    model_fp32, [['conv', 'relu']]
)
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# quantization algorithm calibration happens here
# this example uses just a single sample, but obvious in prod you will
# want to use some meaningful subset of your training or test set
# instead.
input_fp32 = torch.randn(4, 1, 4, 4)
model_fp32_prepared(input_fp32)

model_int8 = torch.quantization.convert(model_fp32_prepared)
res = model_int8(input_fp32)
```

Here's a couple more things you need to keep in mind:

- Static quantization requires inserting `QuantStub` and `DeQuantStub` layers into the model module initialization code. When the model in question is a pretrained one from somewhere else, e.g. `huggingface`, this is non-trivial to do; `torchvision` and `huggingface` are starting to release their own prequantized versions of their models for exactly this reason.
- Static quantization requires a calibration pass on a CPU device using the same (supported) instruction set as the deployment target.

In practice, static quantization is the right technique for medium-to-large sized models making heavy use of convolutions. [PyTorch's own best-effort benchmarks](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#device-and-operator-support) use static quantization more than they do the other two techniques.

## Quantization-aware training

The final, most accurate, but also most tedious quantization technique is **quantization-aware training** (QAT). QAT does away with the post-training calibration process static quantization uses by injecting it into the training process directly.

QAT works by injecting `FakeQuantile` layers into the model, which simulates `int8` behavior in `fp32` at training time by scaling and rounding their inputs. This behavior, which occurs during both forward and backpropagation, makes the model optimizer itself aware of the quantization behavior.

Injecting quantization into model optimization directly like this leads to the best performance, but it also requires (potentially significant, potentially very significant) model fine-tuning to ensure that the model continues to converge. It also slows down training time.

Aside from that, the QAT API looks almost exactly like the static quantization API, with the exception that the methods are now prefixed or affixed `qat`:

```python
# not eval!
model_fp32.train()
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_fp32_fused = torch.quantization.fuse_modules(model_fp32,
    [['conv', 'bn', 'relu']])
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused)

# calibration
training_loop(model_fp32_prepared)

model_fp32_prepared.eval()
model_int8 = torch.quantization.convert(model_fp32_prepared)
```

Since the API is almost exactly the same, we will omit further discussion of it here.

The PyTorch team found that, in practice, QAT is only necessary when working with very heavily optimized convolutional models, e.g. MobileNet, which have very sparse weights. As such, QAT is potentially a useful technique for edge deployments, but should not be necessary for server-side deployments. To learn more, refer to their blog post [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/).

## Benchmarks

To test the effect that model quantization has in practice, I ran each technique on an example model for which it is a good fit (links point to the model code—I recommend giving the code a quick skim):

For dynamic quantization, [a Twitter sentiment classifier with a BERT backbone](https://github.com/spellml/tweet-sentiment-extraction/blob/master/servers/eval_quantized.py).
For static quantization, [a UNet semantic image classifier trained on Bob Ross images](https://github.com/spellml/unet-bob-ross/blob/master/servers/eval_quantized.py).
For quantization-aware training, [a MobileNet trained on CIFAR10](https://github.com/ResidentMario/mobilenet-cifar10/blob/master/servers/eval_quantized.py).

In this section I present some benchmarks from some experiments I ran using these three models.

To begin, I trained each model, then scored it on its training dataset. I ran scoring jobs on GPU, CPU without quantization, and CPU with quantization. GPU inference was carried out using an NVIDIA T4 instance (g4dn.xlarge) on AWS; CPU inference was carried out using a c5.4xlarge (a medium-sized CPU instance type). The jobs were executed using a [Spell run](https://spell.ml/docs/run_overview/), using commands like this one:

```bash
$ spell run
    --github-url https://github.com/spellml/mobilenet-cifar10.git \
    --machine-type cpu-big \
    --mount runs/480/checkpoints/model_10.pth:\
            /spell/checkpoints/model_10.pth python servers/eval.py
```

Here are the results:

![Quant time benchmarks](/img/ch5/quant-time-benchmarks.avif)

Looking at these results, we can see that GPU inference still beats quantized CPU inference handedly. However, quantization goes a long way towards closing this performance gap, providing speedups of 30 to 50 percent.

Next, let's take a look at the effect that quantization has on model size by measuring its footprint on disk:

![Quant size benchmarks](/img/ch5/quant-size-benchmarks.avif)

The statically quantized and QAT models demonstrate the "approaching 75%" model size reduction I alluded to earlier in this article. Meanwhile, dynamic quantization does not affect the size of the model on disk—the model is still read from and saved to disk in fp32, so no savings there.

As you can see, quantization is a powerful technique for reducing model inference time on CPUs—and hence, a key component to making model inference, both on CPU compute and on edge devices, computationally tractable. If you're thinking of making use of quantization, some other techniques important to this space, like [model distillation](https://heartbeat.fritz.ai/research-guide-model-distillation-techniques-for-deep-learning-4a100801c0eb) and [pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html), are also worth exploring.

## To-do

- Re-do the benchmarks
- Has anything changed in this part of the API?
