import multiprocessing as mp
from functools import partial
import pandas as pd
import numpy as np
from typing import Callable, Dict, List
from .dataset_templates import DATASET_TEMPLATES

def generate_dataset_chunk(
    template_func: Callable,
    chunk_size: int,
    random_seed: int
) -> pd.DataFrame:
    """
    Generate a chunk of dataset using the specified template function.
    
    Args:
        template_func: The dataset generation function
        chunk_size: Number of samples in this chunk
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing the generated data
    """
    np.random.seed(random_seed)
    return template_func(n_samples=chunk_size)

def parallel_generate_dataset(
    template_name: str,
    total_samples: int,
    n_workers: int = None
) -> pd.DataFrame:
    """
    Generate a dataset in parallel using multiple processes.
    
    Args:
        template_name: Name of the dataset template to use
        total_samples: Total number of samples to generate
        n_workers: Number of worker processes (default: number of CPU cores)
        
    Returns:
        DataFrame containing the generated data
    """
    if template_name not in DATASET_TEMPLATES:
        raise ValueError(f"Template '{template_name}' not found")
        
    template_func = DATASET_TEMPLATES[template_name]
    
    # Use all available CPU cores if n_workers not specified
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Calculate chunk sizes
    chunk_size = total_samples // n_workers
    remaining = total_samples % n_workers
    
    # Create pool of workers
    with mp.Pool(processes=n_workers) as pool:
        # Generate random seeds for each chunk
        random_seeds = np.random.randint(0, 10000, size=n_workers)
        
        # Prepare arguments for each chunk
        chunk_args = []
        for i in range(n_workers):
            # Add remaining samples to the last chunk
            current_chunk_size = chunk_size + (remaining if i == n_workers - 1 else 0)
            chunk_args.append((template_func, current_chunk_size, random_seeds[i]))
        
        # Generate chunks in parallel
        chunks = pool.starmap(generate_dataset_chunk, chunk_args)
    
    # Combine chunks into final dataset
    return pd.concat(chunks, ignore_index=True)

def parallel_generate_multiple_datasets(
    template_names: List[str],
    total_samples: int,
    n_workers: int = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate multiple datasets in parallel.
    
    Args:
        template_names: List of dataset template names to generate
        total_samples: Total number of samples for each dataset
        n_workers: Number of worker processes per dataset
        
    Returns:
        Dictionary mapping template names to generated datasets
    """
    results = {}
    for template_name in template_names:
        results[template_name] = parallel_generate_dataset(
            template_name,
            total_samples,
            n_workers
        )
    return results 