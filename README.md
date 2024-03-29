
# K.I.T.T.

This repository is the official implementation of Kernel Identification Through Transformers (https://arxiv.org/abs/2106.08185), presented at [NeurIPS 2021](https://neurips.cc/Conferences/2021). 

This project aims to rapidly identify suitable expressive kernels for a given dataset.
Motivated by the success of image captioning architectures, we adopt a two-stage approach. 
First a classifier is trained to identify primitive kernels, 
The features generated by the classifier can be fed into a second network which assembles a more complex caption. 

## Installation

Ensure poetry is installed and run:

`poetry env use python3.7`

`poetry install`

from the top level directory of this repo.


## Running

To make use of KITT, two networks need to be trained.

First a classifier: \
[kitt.prototype.scripts.train_classifier.py](https://github.com/Prowler-io/kitt/blob/master/kitt/prototype/scripts/train_classifier.py)

Secondly, the captioning network: \
[kitt.prototype.scripts.train_kitt.py](https://github.com/Prowler-io/kitt/blob/master/kitt/prototype/scripts/train_kitt.py)

Once these networks are trained, several experiments can run from: \
[kitt.prototype.scripts.experiments](https://github.com/Prowler-io/kitt/tree/master/kitt/prototype/scripts/experiments)

Several scripts make use of sacred. 
An introduction to sacred can be found [here](https://sacred.readthedocs.io/en/stable/quickstart.html#hello-world).


## Testing

From the root directory of this repo, run:

`poetry run task test`

## Relevant references
* [Task-Agnostic Amortized Inference of Gaussian Process Hyperparameters](https://github.com/PrincetonLIPS/AHGP)
* [The Meshed-Memory Transformer](https://github.com/aimagelab/meshed-memory-transformer)
* [The Set Transformer](https://github.com/arrigonialberto86/set_transformer)
* [Deep Sets](https://arxiv.org/abs/1703.06114)
* [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)
