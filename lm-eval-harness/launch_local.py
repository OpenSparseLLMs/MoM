
import os
import torch
import click

from typing import List, Optional

from lm_eval.__main__ import cli_evaluate
from datetime import datetime
import os
import importlib.util

from tqdm import tqdm

MAX_WORKERS_PER_GPU = 1
DEVICE = "cuda"
torch.random.manual_seed(0)


def execute_config(
    model: str,
    task: str,
    batch_size: int,
    limit: int,
    output_dir: str,
    num_fewshot: int,

    context_length: int = 1000,
    answer_length: int = 50,
    cutting_context: bool = False,
    decode_mode: str = 'default',
):
    # Save the original standard output
    import subprocess

    output_dir = os.path.join(output_dir, model, task)

    # pass flags to cli_evaluate() to override the defaults in argparse
    args = [
            "lm_eval",
            "--model", "lm_eval_model",
            "--model_args", f" checkpoint_name={model}", 
            "--task", task,
            "--device", "cuda:0",
            "--batch_size", str(batch_size),
            "--output_path", output_dir,
            "--num_fewshot", str(num_fewshot),
            "--decode_mode", decode_mode,
            
            "--log_samples",
            "--write_out",
            # "--output", f"{run_name}/",
    ]

    if cutting_context:
        args.extend(["--cutting_context"])
        args.extend(["--context_length", str(context_length)])
        args.extend(["--answer_length", str(answer_length)])
        args.extend(["--context_key", "text"])
    
    if limit is not None:
        args.extend(["--limit", str(limit)])
    
    subprocess.run(args)            

    print(f"Decoded with mode: {decode_mode}")   


@click.command()
@click.option("-m", "--model", type=str, multiple=True)
@click.option("-t", "--task", type=str, multiple=True)
@click.option("-p", "--parallelize", is_flag=True)
@click.option("--gpus", default=None, type=str)
@click.option("--batch-size", default=8, type=int)
@click.option("--limit", default=None, type=int)
@click.option("--num_fewshot", default=0, type=int)
@click.option("--context_length", default=1000, type=int)
@click.option("--answer_length", default=50, type=int)
@click.option("--output_dir", default="output", type=str)
@click.option("--cutting_context", is_flag=True)
@click.option("--decode_mode", default="default", type=str)
def main(
    model: List[str], 
    task: List[str], 
    batch_size: int,
    limit: Optional[int],
    parallelize: bool, 
    gpus: str,
    num_fewshot: int = 0,
    context_length: int = 1000,
    answer_length: int = 50,
    cutting_context: bool = False, 
    output_dir: str = "output",
    decode_mode: str = 'default'
):
    
    if limit is not None and limit < 0: limit = None

    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Load the given Python file as a module
    configs = [
        {"model": m, "task": t} for m in model for t in task
    ]

    use_ray = parallelize and len(configs) > 0
    if use_ray:
        import ray
        # ray was killing workers due to OOM, but it didn't seem to be necessary 
        os.environ["RAY_memory_monitor_refresh_ms"] = "0"
        ray.init(ignore_reinit_error=True, log_to_driver=True)

    print(f"Running sweep with {len(configs)} configs")

    output_dir = f"{output_dir}/{datetime.now().strftime('%y-%m-%d_%H-%M')}"

    # Run each script in parallel using Ray
    if not use_ray:
        for config in configs: 
            execute_config(
                **config,
                batch_size=batch_size,
                limit=limit,
                output_dir=output_dir,
                num_fewshot=num_fewshot,
                context_length=context_length,
                answer_length=answer_length,
                cutting_context=cutting_context,
                decode_mode=decode_mode
            )
    else:
        completed = 0
        total = len(configs)
        print(f"Completed: {completed} ({completed / total:0.1%}) | Total: {total}")

        remote = ray.remote(num_gpus=(1 // MAX_WORKERS_PER_GPU))(execute_config)
        futures = [remote.remote(
            **config, batch_size=batch_size, limit=limit, output_dir=output_dir, 
            num_fewshot=num_fewshot, context_length=context_length, answer_length=answer_length, cutting_context=cutting_context,
            decode_mode=decode_mode
        ) for config in configs]
        
        while futures:
            complete, futures = ray.wait(futures)
            completed += len(complete)
            print(f"Completed: {completed} ({completed / total:0.1%}) | Total: {total}")

        ray.shutdown()
    

if __name__ == "__main__":
    main()

    