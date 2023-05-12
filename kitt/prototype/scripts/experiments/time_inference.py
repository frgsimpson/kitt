from argparse import ArgumentParser

import tensorflow as tf
import pandas as pd
from itertools import product
from kitt.config import BASE_PATH, LOG_DIR
from kitt.utils.save_load_models import load_kitt
from kitt.utils.timing import time_inference


def build_inference_fn(args):
    model = load_kitt(
        args.model_load_path,
        args.encoder,
        args.decoder,
        args.max_products,
        args.max_expression_length
    )

    @tf.function
    def do_inference(data: tf.Tensor):
        return model(data)

    return do_inference


def main(args):
    with tf.profiler.experimental.Profile(
            str(LOG_DIR / "profiling"),
            tf.profiler.experimental.ProfilerOptions(python_tracer_level=1)
    ):
        inference_fn = build_inference_fn(args)
        mean, std = time_inference(inference_fn, args.num_dims, args.seq_length, args.repeats)
    print(f"KITT inference takes {mean} (std {std}) seconds.")
    print(
        f"{args.repeats} repetitions with {args.num_dims} dimensions and "
        f"{args.seq_length} data points."
    )
    return mean, std
    
def timing_experiment(args):
    
    seq_lengths = [100,500,1000,2000]
    dims = [2,5,7,10,15,30,40]
    
    mean_sec = []
    std_sec = []
    inference_fn = build_inference_fn(args)
    for s,d in product(seq_lengths, dims):
            args.seq_length = s
            args.num_dims = d
            mean, std = time_inference(inference_fn, args.num_dims, args.seq_length, args.repeats)
            mean_sec.append(mean)
            std_sec.append(std)
            print(f"KITT inference takes {mean} (std {std}) seconds." + '\n' + 
                  f"{args.repeats} repetitions with {args.num_dims} dimensions and " + 
                  f"{args.seq_length} data points.")
    

    timings = pd.DataFrame(columns=['seq','dims','mean','std'])
    
    timings['seq'] = list(zip(*list(product(seq_lengths, dims))))[0]
    timings['dims'] = list(zip(*list(product(seq_lengths, dims))))[1]
    
    timings['mean'] = mean_sec.numpy()
    timings['std'] = std_sec.numpy()
    return timings
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--encoder", type=str, default="classifier-transformer")
    parser.add_argument("--decoder", type=str, default="decoder-transformer")
    parser.add_argument("--model_load_path", type=str, default=BASE_PATH / "saves" / "models")
    parser.add_argument("--max_products", type=int, default=3)
    parser.add_argument("--max_expression_length", type=int, default=4)
    parser.add_argument("--num_dims", type=int, default=10)
    parser.add_argument("--seq_length", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()

    timings = timing_experiment(arguments)