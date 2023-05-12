#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate ground truth files.

The aggregated files are segregated based on the underlying model used for 
prediction.

For example:
    
If the underlying model was a classification transformer only provide --model_name:

"classifier-transformer_2d_max_exp_1_prod_1_random_inputs"

which indicates that the underlying model was trained on 2d, max_products = 1 and
random inputs. 

Possible options are:

"classifier-transformer_2d_max_exp_1_prod_1_random_inputs"
"classifier-transformer_2d_max_exp_1_prod_1_grid_inputs"
"classifier-transformer_2d_max_exp_1_prod_2_random_inputs"
"classifier-transformer_2d_max_exp_1_prod_2_grid_inputs"
"classifier-transformer_2d_max_exp_1_prod_3_random_inputs"
"classifier-transformer_2d_max_exp_1_prod_3_grid_inputs" 

(x2 for 4d trained options)

If the underlying model was full kitt, provide model name:
    
"kitt_2d_max_exp_1_prod_1_random_inputs"
  
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from kitt.config import LOG_DIR
from argparse import ArgumentParser, Namespace

plt.style.use('ggplot')

def parse_command_line_args() -> Namespace:
    """ As it says on the tin """
    parser = ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the underlying model which made predictions",
        default='classifier-transformer_64_4d_max_exp_1_prod_2_random_inputs'
    )

    parser.add_argument(
        "--test_coords",
        type=str,
        help="Whether the inference was on random inputs or grid inputs",
        default="random"
        )
    parser.add_argument(
        "--dates",
        type=str,
        nargs="*",
        help=f"Name of the log directories containing experimental results, format is e.g. Feb12."
             f"Specify more than one in case the experiment run overnight.",
        default="May16"
    )
    
    return parser.parse_args()

def standard_error(x):
    return np.std(x)/np.sqrt(len(x))

def plot_triple_errorbar_chart(pred_by_dim,
                               pred_by_input_size,
                               pred_by_kernel,
                               test_coords,
                               train_coords,
                               train_dim,
                               train_inputs):
    
   plt.figure(figsize=(14,5))   
   plt.subplot(131)
   plt.errorbar(pred_by_dim.index.array, pred_by_dim['top_1']['mean'], 
                  yerr=pred_by_dim['top_1']['standard_error'], capsize=5, color='b', marker='o', label='Top 1', alpha=0.6)
   plt.errorbar(pred_by_dim.index.array, pred_by_dim['top_3']['mean'], 
                  yerr=pred_by_dim['top_3']['standard_error'], capsize=5, color='r', marker='o', label='Top 3', alpha=0.6)
   plt.errorbar(pred_by_dim.index.array, pred_by_dim['top_5']['mean'], 
                  yerr=pred_by_dim['top_5']['standard_error'], capsize=5, color='g', marker='o', label='Top 5', alpha=0.6)
   plt.xlabel('Test dimension', fontsize='small') 
   plt.title('Robustness to higher dimensions', fontsize='small')
   plt.ylabel('Test Accuracy', fontsize='small')
   
   
   plt.subplot(132)
   plt.errorbar(pred_by_input_size.index.array, pred_by_input_size['top_1']['mean'],
                  yerr=pred_by_input_size['top_1']['standard_error'], capsize=5, color='b', marker='o', label='Top 1', alpha=0.6)
   plt.errorbar(pred_by_input_size.index.array, pred_by_input_size['top_3']['mean'],
                  yerr=pred_by_input_size['top_3']['standard_error'], capsize=5, color='r', marker='o', label='Top 3', alpha=0.6)
   plt.errorbar(pred_by_input_size.index.array, pred_by_input_size['top_5']['mean'],
                  yerr=pred_by_input_size['top_5']['standard_error'], capsize=5, color='g', marker='o', label='Top 5', alpha=0.6)
   plt.xscale('log')
   plt.xlabel('Number of test points', fontsize='small') 
   plt.legend(fontsize='small')
   plt.title('Robustness to test size', fontsize='small')
   plt.ylabel('Test Accuracy', fontsize='small')

   plt.subplot(133)
   labels = pred_by_kernel.index.array
   x = np.arange(len(labels))  # the label locations
   width = 0.35  # the width of the bars
   
   plt.bar(x - width/2, pred_by_kernel['top_3']['mean'].values, width, 
       label='Top 3', color='b', alpha=0.7, 
       yerr=pred_by_kernel['top_3']['standard_error'], capsize=5)
   plt.bar(x + width/2, pred_by_kernel['top_5']['mean'].values, width, 
       label='Top 5', color='r', alpha=0.5, 
       yerr=pred_by_kernel['top_5']['standard_error'], capsize=5)
    
   plt.title('Accuracy by kernel type', fontsize='small')
   plt.ylabel('Test Accuracy', fontsize='small')
   plt.xticks(x, labels, fontsize='x-small')
   plt.legend(fontsize='small')
   plt.suptitle('Classifier Transformer identifying primitive kernel structure' +  '\n' \
               + f'[Train params: train_dim={train_dim}, num train={train_inputs}, train_coords={train_coords}, test_coords={test_coords}]', fontsize='small')
   plt.show()

if __name__ == "__main__":
    
     ## Read in raw results files
    
     args = parse_command_line_args()
     
     data = []
     prefix = 'top_k_accuracy_'
     encoder_name = args.model_name + f'_{args.test_coords}_test.json'
     filename = prefix + encoder_name
     results_file = LOG_DIR / args.dates / 'gtr_results' / f'{filename}'
     with open(results_file) as f:
         predictions = json.loads("[" + 
        f.read().replace("}\n{", "},\n{") + "]")
     
     ## Process the data

     predictions = pd.DataFrame.from_dict(predictions)
    
     ## Aggregation across runs, kernels and dims 
     
     pd.set_option('display.max_columns', None)
     pred_by_input_size = predictions.groupby(['num_train'])[['top_1','top_3','top_5']].agg(['mean', standard_error])
     pred_by_dims = predictions.groupby(['dimension'])[['top_1','top_3','top_5']].agg(['mean', standard_error])
     pred_by_kernel = predictions.groupby(['kernel'])[['top_1','top_3','top_5']].agg(['mean', standard_error])

     plot_triple_errorbar_chart(
         pred_by_dims,
         pred_by_input_size,
         pred_by_kernel,
         test_coords=args.test_coords,
         train_coords='random',
         train_dim=4,
         train_inputs=64)

