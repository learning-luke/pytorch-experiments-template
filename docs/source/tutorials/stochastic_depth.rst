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

1. Implement a stochastic depth block for ResNets
2. Add some flags to the training file for experiment management
3. Train 50, 110 layer ResNets on CIFAR10, CINIC10
4. Generate code to run multiple seeds of each experiment
5. Write simple plotting/table generation code

Hyperparameters
~~~~~~~~~~~~~~~
From the paper:

  The baseline ResNet is trained with SGD for 500 epochs, with a mini-batch size 128. The initial learning rate is 0.1, and is divided by a factor of 10 after epochs 250 and 375. We use a weight decay of 1e-4, momentum of 0.9, and Nesterov momentum [33] with 0 dampening, as suggested by [34]. For stochastic depth, the network structure and all optimization settings are exactly the same as the baseline. All settings were chosen to match the setup of He et al. [8].

Implementation
~~~~~~~~~~~~~~

**1. Implementing a stochastic depth block.**

Here's a relatively straightforward implementation:

.. code-block:: python

  class StochasticDepthBlock(nn.Module):
      def __init__(self, block, stoch_depth_probability=None):
          super(StochasticDepthBlock, self).__init__()
          self.block = block
          self.stoch_depth_probability = torch.Tensor([stoch_depth_probability])

      def forward(self, x):
          if self.training:
              if torch.bernoulli(self.stoch_depth_probability):
                  return self.block.forward(x)
              else:
                  return self.block.shortcut(x)
          else:
              return self.block.forward(x)

We pass in an instantiated block (either `BasicBlock` or `Bottleneck`), and then the StochasticDepthBlock is going to use the Bernoulli variable to decide whether to skip the block or not.

So now `_make_layer` looks like this:

.. code-block:: python

  def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layer = block(self.in_planes, planes, stride)
            if self.stoch_depth_probability is not None:
                layer = StochasticDepthBlock(layer, self.stoch_depth_probability)
            layers.append(layer)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

And the init function looks like this:

.. code-block:: python

  class ResNet(nn.Module):
      def __init__(
          self,
          block,
          num_blocks,
          num_classes=10,
          variant=None,
          in_channels=3,
          stoch_depth_probability=None,
      ):
      self.stoch_depth_probability = stoch_depth_probability
      # rest of init function below


**2. Adding handlers to the train file**

The first step is to add an argument to the `ArgParser`:

.. code-block:: python

  parser.add_argument("--model.stoch_depth_probability", type=float, default=None)

Which later gets passed to the model:

.. code-block:: python

  model = get_model(
        model_type=args.model.type,
        num_classes=num_classes,
        dataset_name=args.dataset_name,
        stoch_depth_probability=args.model.stoch_depth_probability,
    ).to(device)

The second step is to implement a scheduler for the stochastic depth probability. In the paper they schedule it to start at 1.0 and linearly reduce to 0.5 throughout training.

Since this seems like a training hyperparameter, I'll implement this in the train file:

.. code-block:: python

  if args.model.stoch_depth_probability is not None:

        class StochasticDepthProbabilitySchedulerLinear:
            def __init__(
                self,
                stoch_depth_probability=1.0,
                final_stoch_depth_probability=0.5,
                cur_epoch=0,
                max_epochs=args.max_epochs,
                model=None,
            ):
                self.stoch_depth_probability = stoch_depth_probability
                self.schedule = torch.linspace(
                    stoch_depth_probability, final_stoch_depth_probability, max_epochs
                )
                self.cur_epoch = cur_epoch

            def step(self):
                new_stoch_depth_probability = self.schedule[self.cur_epoch]

                for module in model.modules():
                    if isinstance(module, StochasticDepthBlock):
                        module.stoch_depth_probability = new_stoch_depth_probability

                self.cur_epoch += 1

        stoch_depth_probability_scheduler = StochasticDepthProbabilitySchedulerLinear(
            stoch_depth_probability=1.0,
            final_stoch_depth_probability=0.5,
            cur_epoch=0,
            max_epochs=args.max_epochs,
            model=model,
        )
