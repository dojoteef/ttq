# Trained Ternary Quantization

## Background

This is a replication of the Trained Ternary Quantization paper. I looked at
quantization around a year ago, and it seems that the state of the art has
advanced since I last looked at this area of research. TensorFlow has some
methods already built-in for the purpose of quantizing networks after training.
They basically depend on taking a fully trained network and reducing the bits
required for encoding the floating point weights/parameters of the networks as
8-bit fixed point. More recent avenues of research have looked at quantizing
the networks into binary or ternary values.

Relevant papers I researched while working considering the issue of
quantization:  
https://arxiv.org/pdf/1609.07061.pdf  
https://arxiv.org/pdf/1603.05279.pdf  
https://arxiv.org/pdf/1612.01064.pdf  

The final paper, on Trained Ternary Quantization, is the paper I chose to
replicate, as it has the most startling results. Namely, despite quantizing the
network into a 2-bit representation, in somes cases it still achieves the same
accuracy, and can even exceed the accuracy of the original network. This result
seems quite amazing, but it could be partly due to quantization being
considered a form of regularization, which is known to reduce overfitting and
make models generalize better.

As of now, the results of using Trained Ternary Quantization in this project
only allow for a theoretical reduction in computation and parameter size. The
TensorFlow framework does not have built-in operations which support inference
on models that are 2-bit. In order to fully make use of this approach, the
operations and kernels utilized by the model during inference need to be
written to support targeted hardware architectures. More information regarding
discussion of the topic in regards to TensorFlow can be found here:

https://github.com/tensorflow/tensorflow/issues/1592

Some additional thoughts:
 * Similar to the fixed point quanization approach that TensorFlow currently
   supports, Trained Ternary Quantization utilizes an asymmetrical bound for
   the maximum/minimum values of the quantization. This seems to allow for more
   optimally trained model than previous ternary approaches to the problem
   which used a symmetric bound.
 * The effectiveness of these approaches, both binarization and ternarization,
   seem related to the effectiveness of both ReLU and HardTanh respectively, as
   activation functions.

## Results

I chose to use the Resnet family of models as they have good classification
performance compared to the size/speed of the model for inference. The
following tables contain the results of the various tests that were run. The
results are not as strong as I would have expected, which I believe is mainly
due to overfitting; regularization was causing some issues that I have not
ironed out yet.

**NOTE:**
*Parameters* are in **millions**, *Memory usage* is in **MB**, and for the
quantization results the *Memory usage* is the theoretical usage based on the
discussion above (i.e., weights are quantized to 2-bits, except for the first
and last layer; the beta/gamma of the batch norm are 32-bit floats).

_**Without** quantization_

| Model     |  Dataset | Parameters | Memory usage | Accuracy |
|:----------|---------:|-----------:|-------------:|---------:|
|  Resnet18 |    MNIST |       5.60 |        21.33 |   98.45% |
|  Resnet34 |    MNIST |      11.22 |        42.81 |   97.71% |
|  Resnet18 | CIFAR-10 |       5.61 |        21.33 |     TODO |

_**With** quantization_

| Model     |  Dataset | Parameters | Memory usage | Accuracy |
|:----------|---------:|-----------:|-------------:|---------:|
|  Resnet18 |    MNIST |       5.60 |         1.39 |   96.02% |
|  Resnet34 |    MNIST |      11.22 |         2.86 |     TODO |
|  Resnet18 | CIFAR-10 |       5.61 |         1.39 |     TODO |
