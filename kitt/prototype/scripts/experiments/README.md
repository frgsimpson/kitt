## Experiment Scripts

This folder contains scripts used to evaluate trained models in various ways.
These scripts are as follows.

[`evaluate_classifier_generated_models.py`](evaluate_classifier_generated_models.py)
evaluates a trained classifier by instantiating each kernel in the vocabulary
sampling data from it, passing it through the classifier, building the model(s)
proposed by the classifier and training the proposal(s) and the original (generating)
kernel on the original data. The results are nlpd and rmse on the original sample.
This goes beyond ground truth recall and analyses how well the models produced
fit the data.

[`evaluate_kitt.py`](evaluate_kitt.py) performs the same process as `evaluate_classifier_generated_models.py`
but with the full KITT model.

[`ground_truth_experiment.py`](ground_truth_experiment.py) evaluates KITT in terms of
ground truth recall. This script generates plots including the top-3 and top-5 accuracy.

[`run_on_uci.py`](run_on_uci.py) loads the UCI datasets and runs either KITT or a trained
classifier on each split saving the predicted kernels and their probabilities to disk (so
that they can later be instantiated as models and trained using `run_multi_train_parallel.py`).

[`run_multi_train_parallel.py`](run_multi_train_parallel.py) reads in the output of 
`run_on_uci.py` and then instantiates the kernels in GP regression models which are then
trained on the UCI data and rmse and nlpd values reported. The building and training of GPs
is run in parallel and the results written to disk to later be aggregated by `aggregate_results.py`.

[`time_inference.py`](time_inference.py) estimates the time for a single forward pass through
KITT using random values as data.

[`train_full_vocab_on_uci.py`](train_full_vocab_on_uci.py) runs each kernel in a considered
vocabulary against UCI datasets. In future [`run_multi_train_parallel.py`](run_multi_train_parallel.py)
could be expanded to do this task.

[`aggregate_results.py`](aggregate_results.py) aggregates the results over multiple splits
of the UCI datasets. This script reads the outputs of `run_multi_train_parallel.py` and produces
summarised (aggregated) results ready for the manuscript.


### Deprecated Scripts
In the past we had certain scripts which have now been superseded by the scripts
above. A description of how to attain the same functionality is provided below.

[`regression_experiment.py`](https://github.com/Prowler-io/kitt/blob/e0b4af5f3a1dff1272d8637d47caa563de0a9ae8/kitt/prototype/scripts/experiments/regression_experiment.py)
and [`multi_train_regression.py`]() load a single split of a single UCI experiment, run it through a classifier or KITT to generate model
predictions which are then instantiated and trained on the UCI data. These scripts then report NLPD and
RMSE for the test set. This functionality is replaced by [`run_on_uci.py`](run_on_uci.py)
and [`run_multi_train_parallel.py`](run_multi_train_parallel.py) which passes UCI data through KITT or
a classifier and builds and trains GP regression models respectively. The reason for splitting
the script in two is to allow for training GP models in parallel which enables us to scale the
experiments and run them faster at scale. The cost of this is that you need to run two
scripts rather than one in the simple case of running on a single split of a single
UCI dataset.

[`load_demo.py`](https://github.com/Prowler-io/kitt/blob/e0b4af5f3a1dff1272d8637d47caa563de0a9ae8/kitt/prototype/scripts/experiments/load_demo.py) simply provided an example of loading the UCI data which can now be seen in 
several of the existing scripts (e.g. [`run_on_uci.py`](run_on_uci.py)).

[`rbf_ard_benchmark.py`](https://github.com/Prowler-io/kitt/blob/e0b4af5f3a1dff1272d8637d47caa563de0a9ae8/kitt/prototype/scripts/experiments/rbf_ard_benchmark_uci.py) performed a baseline experiment where an RBF kernel was trained on 
the UCI datasets with ARD. This functionality is now possible in [`run_multi_train_parallel.py`](run_multi_train_parallel.py)
by passing the `--rbf_baseline_experiment` argument.