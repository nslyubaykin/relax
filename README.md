# ReLAx
ReLAx - Reinforcement Learning Applications

Relax is an object oriented library for deep reinforcement learning built on top of PyTotch.

<!-- toc -->

- [Header1](#header1)
- [Header2](#header2)
- [Header3](#header3)
- [Header4](#header4)
- [Header5](#header5)
- [Header6](#header6)
- [Header7](#header7)
- [Header8](#header8)
- [Header9](#header9)

<!-- tocstop -->

## Header1

## Header2

## Header3


- [More About PyTorch](#more-about-pytorch)
  - [A GPU-Ready Tensor Library](#a-gpu-ready-tensor-library)
  - [Dynamic Neural Networks: Tape-Based Autograd](#dynamic-neural-networks-tape-based-autograd)
  - [Python First](#python-first)
  - [Imperative Experiences](#imperative-experiences)
  - [Fast and Lean](#fast-and-lean)
  - [Extensions Without Pain](#extensions-without-pain)
- [Installation](#installation)
  - [Binaries](#binaries)
    - [NVIDIA Jetson Platforms](#nvidia-jetson-platforms)
  - [From Source](#from-source)
    - [Prerequisites](#prerequisites)
    - [Install Dependencies](#install-dependencies)
    - [Get the PyTorch Source](#get-the-pytorch-source)
    - [Install PyTorch](#install-pytorch)
      - [Adjust Build Options (Optional)](#adjust-build-options-optional)
  - [Docker Image](#docker-image)
    - [Using pre-built images](#using-pre-built-images)
    - [Building the image yourself](#building-the-image-yourself)
  - [Building the Documentation](#building-the-documentation)
  - [Previous Versions](#previous-versions)
- [Getting Started](#getting-started)
- [Resources](#resources)
- [Communication](#communication)
- [Releases and Contributing](#releases-and-contributing)
- [The Team](#the-team)
- [License](#license)

<!-- tocstop -->

## More About PyTorch

At a granular level, PyTorch is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | A Tensor library like NumPy, with strong GPU support |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | A tape-based automatic differentiation library that supports all differentiable Tensor operations in torch |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html) | A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code  |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | A neural networks library deeply integrated with autograd designed for maximum flexibility |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) | DataLoader and other utility functions for convenience |

Usually, PyTorch is used either as:

- A replacement for NumPy to use the power of GPUs.
- A deep learning research platform that provides maximum flexibility and speed.

Elaborating Further:

### A GPU-Ready Tensor Library

If you use NumPy, then you have used Tensors (a.k.a. ndarray).

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

PyTorch provides Tensors that can live either on the CPU or the GPU and accelerates the
computation by a huge amount.

We provide a wide variety of tensor routines to accelerate and fit your scientific computation needs
such as slicing, indexing, math operations, linear algebra, reductions.
And they are fast!
