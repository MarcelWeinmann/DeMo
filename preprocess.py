from argparse import ArgumentParser
from pathlib import Path
from typing import List
import multiprocessing
from tqdm import tqdm
from src.datamodule.av2_extractor import Av2Extractor

# --- NEW: Setup for multiprocessing workers ---
global_extractor = None

def init_worker(save_dir, mode, num_hist, num_future):
    """
    Called once when a worker process starts.
    Creates a local instance of the extractor so it doesn't need to be pickled.
    """
    global global_extractor
    global_extractor = Av2Extractor(
        save_path=save_dir, 
        mode=mode,
        num_historical_steps=num_hist,
        num_future_steps=num_future
    )

def process_scenario(scenario_file):
    """Top-level function mapped to the pool. Uses the worker's local extractor."""
    return global_extractor.save(scenario_file)
# ----------------------------------------------


def glob_files(data_root: Path, mode: str):
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.parquet"))
    return scenario_files


def preprocess(args):
    batch = args.batch
    data_root = Path(args.data_root)

    for mode in ["train", "val", "test"]:
        save_dir = Path("data/DeMo_processed_sd") / mode
        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode)

        if args.parallel:
            # Set up the pool to initialize the extractor in each worker
            with multiprocessing.Pool(
                processes=16, 
                initializer=init_worker, 
                initargs=(save_dir, mode, args.num_historical_steps, args.num_future_steps)
            ) as p:
                # Use the top-level wrapper function instead of extractor.save
                all_name = list(tqdm(p.imap(process_scenario, scenario_files), total=len(scenario_files)))
        else:
            # If running sequentially, create a single extractor in the main process
            extractor = Av2Extractor(
                save_path=save_dir, 
                mode=mode,
                num_historical_steps=args.num_historical_steps,
                num_future_steps=args.num_future_steps
            )
            for file in tqdm(scenario_files):
                extractor.save(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, default='/path/to/data_root')
    parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--num_historical_steps", type=int, default=50)
    parser.add_argument("--num_future_steps", type=int, default=60)

    args = parser.parse_args()
    preprocess(args)
