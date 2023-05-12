""" Test out the classifier (perhaps expand to support captioning too) on some synthetic data. """
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from kitt.prototype.get_models import get_trained_classification_network, get_trained_captioning_network
from kitt.prototype.evaluate_classifier import infer_top_n_expressions
from kitt.prototype.evaluate_captions import infer_top_n_captions
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.utils.misc import get_args_string
from itertools import product
from joblib import delayed, Parallel
from kitt.data.sampler.utils import (
    load_default_coords, load_random_coords,
    order_sum_kernel_by_variance,
    randomise_hyperparameters,
    make_kernel_from_label,
)
import argparse
from pathlib import Path
from kitt.utils.misc import yes_or_no
import numpy as np
from kitt.config import LOG_DIR
import json
from datetime import datetime

resolutions = [4, 8, 16]
dimensions = [4]
runs = range(2)
batch_size = 20
primitive_vocab = ['LINEAR', 'PERIODIC', 'RBF', 'MATERN52', 'NOISE']
def get_sample(generator, k_string):
    expression = [k_string]  # Full caption
    kernel = make_kernel_from_label(expression, generator.ndims, generator.sigma, generator.all_dims)
    randomise_hyperparameters(kernel)
    kernel = order_sum_kernel_by_variance(kernel)
    return generator.make_batch_of_samples(kernel)

def get_data_cfg_from_encoder_name(encoder_name):
    
    return encoder_name[23:]

def experiment_name(encoder_name, decoder_name):
    
    if decoder_name is None:
        return 'top_k_accuracy_' + encoder_name
    else:
        return 'top_k_accuracy_' + 'kitt_' +  get_data_cfg_from_encoder_name(encoder_name)

def instantiate_sample_generator(test_coords, resolution, test_dimension, decoder, tokenizer):
    
    # all_dims, make_captions and include_x_with_samples are fixed and dont change after init

    generator = SampleGenerator(
    x_values=get_coords(test_coords, resolution, test_dimension),
    x_resolution=resolution,
    batch_size=1,
    min_expression=1,
    max_expression=1,
    tokenizer=tokenizer,
    all_dims=True,  
    make_captions= bool(decoder is not None),
    include_x_with_samples = bool(test_coords == 'random'),
    iterations_per_epoch=1_000)
    
    return generator

def get_coords(test_coords, resolution, test_dimension):
    
    if test_coords == 'random':
        coords = load_random_coords(resolution**2, n_dims=test_dimension) # these values are only for initialisation, 
                                                  # they are updated inside single run 
    else:
        coords = load_default_coords(x_resolution=resolution, y_resolution=resolution)
    return coords

def get_y_values(image, test_coords, input_size):
    
    if test_coords == 'grid':
        return image.reshape(input_size,1)
    else:
        return image[0][:,-1][:,None]
        
    
def get_top_k_batch_accuracy(gtr, pred_across_batch):
    
    pred_top = np.array(pred_across_batch)[:,0]
    pred_top_3 = np.array(pred_across_batch)[:,0:3]
    pred_top_5 = np.array(pred_across_batch)[:,0:5]
    
    batch_size = len(pred_across_batch)
    
    top_1 = np.array(([True if gtr in top else False for top in pred_top])).sum()/batch_size
    top_3 = np.array(([True if gtr in top else  False for top in pred_top_3])).sum()/batch_size
    top_5 = np.array(([True if gtr in top else  False for top in pred_top_5])).sum()/batch_size
    
    return top_1, top_3, top_5

def single_run(
    test_coords,
    run_index: int,
    resolution: int,
    test_dimension: int,
    kernel: str,
    date_str: str,
    encoder, 
    decoder,
    tokenizer,
    filename
  ):
    
    input_size = resolution**2
    print(f'Performing batch predictions for {kernel} input size {input_size} \
           dimension {test_dimension} on {test_coords} coords')
    
    generator = instantiate_sample_generator(test_coords, resolution, test_dimension, decoder, tokenizer)
    pred_across_batch = []
    for batch_index in range(batch_size): 
        image = get_sample(generator, kernel)
        y_values = get_y_values(image, test_coords, input_size)
        if decoder is not None:
            kernel_preds, weights = infer_top_n_captions(encoder, decoder, generator, generator.x_values, y_values, 5)
        else:
            kernel_preds, weights = infer_top_n_expressions(encoder, tokenizer, generator.x_values, y_values, 5)
        
        pred_across_batch.append(kernel_preds)
    # Calculate top-1, top-3 and top-5 accuracy and save
    top_1, top_3, top_5 = get_top_k_batch_accuracy(kernel, pred_across_batch)
    
    results_dict = {'kernel': kernel, 'num_train' : input_size, 'dimension' : test_dimension,
                    'top_1': top_1, 'top_3': top_3, 'top_5': top_5}
    
    with open(filename, "a") as fp:
        json.dump(results_dict, fp, indent=4)
        fp.writelines('\n')
        
def main(args: argparse.Namespace) -> None:
    
    np.random.seed(args.seed)

    date_str = datetime.now().strftime('%b%d')
    save_dir = LOG_DIR / date_str / "gtr_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.decoder_name is None:
        # Use classifier-transformer only
        encoder, tokenizer = get_trained_classification_network(args.encoder_name, max_terms=args.max_terms) 
        decoder = None
    else:
        # Use full-KITT
        encoder, decoder, sample_generator = get_trained_captioning_network(encoder_name=args.encoder_name,
                                                                      decoder_name=args.decoder_name,
                                                                        max_terms=args.max_terms)
    args_save_path = save_dir / "run_gtr_parallel_arguments.txt"
    with open(str(args_save_path), "w") as file:
        file.write(get_args_string(args))

    print("Training GTR in parallel. GPUs may run out of memory, use CPU if this happens. "
        "To use CPU only, set environment variable CUDA_VISIBLE_DEVICES=-1")  

    experiment_filename = experiment_name(args.encoder_name, args.decoder_name)
    results_path = LOG_DIR / date_str / "gtr_results" / experiment_filename
    filename = f"{results_path}_{args.test_coords}_test.json"
    
    if Path(filename).exists():
      overwrite = yes_or_no(f"{filename} will be overwritten. Proceed?")
      if overwrite:
          import os
          os.remove(filename)      
      else:
          return;

    for run_index, resolution, test_dimension, kernel in product(runs, resolutions, dimensions, primitive_vocab):
        single_run(
                args.test_coords,
                run_index,
                resolution,
                test_dimension,
                kernel,
                date_str,
                encoder,
                decoder,
                tokenizer,
                filename) 
    
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=5)
        parser.add_argument("--encoder_name", type=str, default="classifier-transformer_64_4d_max_exp_1_prod_2_random_inputs")# default="classifier-transformer_2d_max_exp_1_prod_1_random_inputs")
        parser.add_argument("--decoder_name", type=str, default=None)
        parser.add_argument("--max_terms", type=int, default=2, help="For the moment must match max_terms used in training.") # see encoder name
        parser.add_argument("--test_coords", type=str, default='random')
        parser.add_argument("--full_kitt", dest="full_kitt", action="store_true")
        parser.add_argument("--classifier_only", dest="full_kitt", action="store_false")
        parser.set_defaults(subset=True, full_kitt=True)
    
        arguments = parser.parse_args()
        main(arguments)
        
