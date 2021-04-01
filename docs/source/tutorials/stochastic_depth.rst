================
Stochastic Depth
================

.. sectnum::

.. contents::

This is a quick tutorial illustrating how to implement stochastic depth in the `pytorch-experiments-template`.

Background
~~~~~~~~~~

`Stochastic depth <https://arxiv.org/abs/1603.09382v1>`_ is a technique for training very deep networks by randomly replacing  entire residual blocks with skip connections during training (a little bit like dropout but at the block level), and then using all of the blocks at test time.

Specifically, for a ResNet block, the input to the current layer :math:`f_{\ell}` is the output of the previous layer :math:`H_{\ell-1})`. We return the output of :math:`f_{\ell}` and skip-connect the output of the previous layer.

In *stochastic depth*, we also add a Bernoulli variable :math:`b_{\ell}` which acts as a binary switch for the layer:

.. math::
  H_{\ell} = \text{ReLU}(b_{\ell} f_{\ell}(H_{\ell-1}) + \text{id}(H_{\ell-1})).

In the paper they set the probability of the Bernoulli distribution to linearly decay from 1 to 0.5 throughout training.

Experiment Plan
~~~~~~~~~~~~~~~

What we'll do in this repository is:

* implement a stochastic depth block for ResNets
* add some flags to the training file for experiment management
* train 50, 110 layer ResNets on CIFAR10, CINIC10
* generate code to run multiple seeds of each experiment
* write simple plotting/table generation code

Hyperparameters
~~~~~~~~~~~~~~~
From the paper:

  The baseline ResNet is trained with SGD for 500 epochs, with a mini-batch size 128. The initial learning rate is 0.1, and is divided by a factor of 10 after epochs 250 and 375. We use a weight decay of 1e-4, momentum of 0.9, and Nesterov momentum [33] with 0 dampening, as suggested by [34]. For stochastic depth, the network structure and all optimization settings are exactly the same as the baseline. All settings were chosen to match the setup of He et al. [8].

Implementation
~~~~~~~~~~~~~~

**1. Implementing a stochastic depth block.**

Since we're modifying the ResNet blocks, I think the best way to do this is to subclass them.

As a reminder, the Bottleneck block looks like this:

.. code-block:: python

  class Bottleneck(nn.Module):
      expansion = 4

      def __init__(self, in_planes, planes, stride=1):
          super(Bottleneck, self).__init__()
          self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
          self.bn1 = nn.BatchNorm2d(planes)
          self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
          self.bn2 = nn.BatchNorm2d(planes)
          self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
          self.bn3 = nn.BatchNorm2d(self.expansion * planes)

          self.shortcut = nn.Sequential()
          if stride != 1 or in_planes != self.expansion * planes:
              self.shortcut = nn.Sequential(
                  nn.Conv2d(
                      in_planes,
                      self.expansion * planes,
                      kernel_size=1,
                      stride=stride,
                      bias=False,
                  ),
                  nn.BatchNorm2d(self.expansion * planes),
              )

      def forward(self, x):
          out = F.relu(self.bn1(self.conv1(x)))
          out = F.relu(self.bn2(self.conv2(out)))
          out = self.bn3(self.conv3(out))
          out += self.shortcut(x)
          out = F.relu(out)
          return out
