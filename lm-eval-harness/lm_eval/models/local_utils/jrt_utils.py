
import os
from typing import List
import pandas as pd
import json
import functools
import tempfile
import wandb


def resolve_user_path(path: str):
    # case on the user using whoami equivalent 
    from getpass import getuser
    user = getuser()
    if user == "eyuboglu@stanford.edu":
        name = "sabri"
    elif user == "simarora@stanford.edu":
        name = "sim"
    else:
        raise ValueError(f"Unknown user {user}")
    return path.format(user=name)

def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever

def unflatten_dict(d: dict) -> dict:
    """ 
    Takes a flat dictionary with '/' separated keys, and returns it as a nested dictionary.
    
    Parameters:
    d (dict): The flat dictionary to be unflattened.
    
    Returns:
    dict: The unflattened, nested dictionary.
    """
    result = {}

    for key, value in d.items():
        parts = key.split('/')
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result

def flatten_dict(data: dict, parent_key:str=None, sep: str='.'):
    """
    Flatten a multi-level nested collection of dictionaries and lists into a flat dictionary.
    
    The function traverses nested dictionaries and lists and constructs keys in the resulting 
    flat dictionary by concatenating nested keys and/or indices separated by a specified separator.
    
    Parameters:
    - data (dict or list): The multi-level nested collection to be flattened.
    - parent_key (str, optional): Used in the recursive call to keep track of the current key 
                                  hierarchy. Defaults to an empty string.
    - sep (str, optional): The separator used between concatenated keys. Defaults to '.'.
    
    Returns:
    - dict: A flat dictionary representation of the input collection.
    
    Example:
    
    >>> nested_data = {
    ...    "a": 1,
    ...    "b": {
    ...        "c": 2,
    ...        "d": {
    ...            "e": 3
    ...        }
    ...    },
    ...    "f": [4, 5]
    ... }
    >>> flatten(nested_data)
    {'a': 1, 'b.c': 2, 'b.d.e': 3, 'f.0': 4, 'f.1': 5}
    """
    items = {}
    if isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key is not None else str(i)
            items.update(flatten_dict(v, new_key, sep=sep))
    elif isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key is not None else k
            items.update(flatten_dict(v, new_key, sep=sep))
    else:
        items[parent_key] = data
    return items



def import_object(name: str):
    """Import an object from a string.
    
    Parameters:
    - name (str): The name of the object to import.
    
    Returns:
    - object: The imported object.
    """
    import importlib
    module_name, obj_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def extended_json_dump(data: any, path: str=None):
    """Extended JSON dump that can handle pydantic models."""
    from pydantic import BaseModel
    import json

    def default(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
    # Save arguments to a JSON file
    if path is not None:
        with open(path, 'w') as f:
            json.dump(data, f, sort_keys=True, default=default)
    else:
        return json.dumps(data, sort_keys=True, default=default)



def hash_args(*args, **kwargs):
    import hashlib
    # Convert the dictionary to a JSON string
    json_str = extended_json_dump({"args": list(args), "kwargs": kwargs})
    # Hash the JSON string using SHA256 (can be replaced with your hash function)
    hash_object = hashlib.sha256(json_str.encode())
    
    # Get the hexdigest which can be used as the hash
    return hash_object.hexdigest()


def cache(cache_dir: str, force: bool = False):
    """
    A decorator that caches the output DataFrame(s) of the decorated function in a directory.
    The directory is determined by hashing the arguments of the function call.
    If the cached output exists for given arguments, it is loaded from the file(s) instead of calling the function.
    The arguments of the function are stored in a separate JSON file in the same directory.
    """
    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            argument_hash = hash_args(*args, **kwargs)
            dir_path = f"{cache_dir}/{argument_hash}"

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            arguments_file = f"{dir_path}/arguments.json"

            # Try to load cached data
            output_dfs = []

            if not force:
                i = 0
                while os.path.exists(f"{dir_path}/dataframe{i}.feather"):
                    output_dfs.append(pd.read_feather(f"{dir_path}/dataframe{i}.feather"))
                    i += 1

            if output_dfs:  
                # If any output was found, return it
                print(f"Loaded cached data from {dir_path}")
                if len(output_dfs) == 1:
                    return output_dfs[0] 
                return tuple(output_dfs)
            else:
                # If no output was found, call the function
                output_dfs = func(*args, **kwargs)

                # Check that the function's return value is a DataFrame or tuple
                if isinstance(output_dfs, pd.DataFrame):
                    output_dfs = (output_dfs,)  # convert single dataframe to tuple
                elif isinstance(output_dfs, tuple) and all(isinstance(df, pd.DataFrame) for df in output_dfs):
                    pass  # all good, it's a tuple of DataFrames
                else:
                    raise ValueError("Function must return a DataFrame or a tuple of DataFrames")

                # Cache the output data
                print(f"Caching data to {dir_path}")
                for i, output_df in enumerate(output_dfs):
                    output_df = output_df.reset_index(drop=True)
                    output_df.to_feather(f"{dir_path}/dataframe{i}.feather")

                # Cache the arguments
                extended_json_dump({"args": list(args), "kwargs": kwargs}, arguments_file)

                if len(output_dfs) == 1:
                    return output_dfs[0]
                return output_dfs

        return wrapper
    return actual_decorator


def log_table(
    df:  pd.DataFrame,
    name: str,
):
    with tempfile.TemporaryDirectory(prefix="/work/sabri_data") as tmp_dir:
        path = os.path.join(tmp_dir, f"{name}.feather")
        df.reset_index(drop=True).to_feather(path)
        artifact = wandb.Artifact(
            f"run-{wandb.run.id}-{name}", type="feather"
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

def load_table(
    artifact_id: str,
    project_name: str = "jrt",
    entity: str = "hazy-research"
) -> pd.DataFrame:
    """
    Fetch a table from wandb based on the artifact ID.
    Args: 
        artifact_id (str): You can find this on the Artifacts tab in the column "Name". 
            For example "run-fsa64p5b-evalpredictionsiter1000:v0".
        project_name (str): Name of the wandb project. Defaults to "zg-neox".
        entity (str): The entity (usually a user or team) in wandb. Defaults to "hazy-research".

    Returns:
        pd.DataFrame: Fetched table in DataFrame format.
    """
    import wandb
    import tempfile
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project_name}/{artifact_id}")

    # download to temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_dir = artifact.download(root=tmp_dir)
        
        # find the table files
        tables = [file for file in os.listdir(artifact_dir) if file.endswith(".table.json") or file.endswith(".feather")]
        
        # validate number of tables found
        assert len(tables) == 1, f"Expected 1 table, found {len(tables)} tables."
        table_path = os.path.join(artifact_dir, tables[0])
        if table_path.endswith(".feather"):
            df = pd.read_feather(table_path)
        else:
            with open(table_path, 'r') as file:
                json_dict = json.load(file)
            df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
    return df 

def load_config(run_id: int):
    """
    Load a config from a wandb run ID.
    Parameters:
        run_id (str): A full wandb run id like "hazy-research/attention/159o6asi"
    """
    # 1: Get configuration from wandb
    api = wandb.Api()
    run = api.run(run_id)
    config = unflatten_dict(run.config)
    return config


def fetch_wandb_runs(project_name: str, filters: dict=None, **kwargs) -> pd.DataFrame:
    """
    Fetches run data from a W&B project into a pandas DataFrame.
    
    Parameters:
    - project_name (str): The name of the W&B project.
    
    Returns:
    - DataFrame: A pandas DataFrame containing the run data.
    """
    # Initialize an API client
    api = wandb.Api()
    
    filters = {} if filters is None else filters
    for k, v in kwargs.items():
        if isinstance(v, List): filters[f"config.{k}"] = {"$in": v}
        else: filters[f"config.{k}"] = v
    
    # Get all runs from the specified project (and entity, if provided)
    runs = api.runs(
        project_name,
        filters=filters
    )
    
    # Create a list to store run data
    run_data = []

    # Iterate through each run and extract relevant data
    for run in runs:
        data = {
            "run_id": run.id,
            "name": run.name,
            "project": run.project,
            "user": run.user.name,
            "state": run.state,
            **flatten_dict(run.config),
            **flatten_dict({**run.summary})
        }
        run_data.append(data)
    
    # Convert list of run data into a DataFrame
    df = pd.DataFrame(run_data)
    df = df.dropna(axis="columns", how="all")

    # can't be serialized
    if "_wandb" in df.columns:
        df = df.drop(columns=["_wandb"])
    if "val_preds" in df.columns:
        df = df.drop(columns=["val_preds"])

    return df