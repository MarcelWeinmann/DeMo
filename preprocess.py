from argparse import ArgumentParser
from pathlib import Path
from typing import List
import multiprocessing
from tqdm import tqdm
from src.datamodule.av2_extractor import Av2Extractor


def glob_files(data_root: Path, mode: str):
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.parquet"))
    return scenario_files


def preprocess(args):
    batch = args.batch
    data_root = Path(args.data_root)

    for mode in args.splits:
        save_dir = Path(args.save_dir) / mode
        extractor = Av2Extractor(save_path=save_dir, mode=mode,
                                 num_historical_steps=args.num_historical_steps,
                                 num_future_steps=args.num_future_steps,
                                 use_raceline=args.use_raceline,
                                 use_raceline_velocity=args.use_raceline_velocity,
                                 speed_cap_column=args.speed_cap_column)

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode)

        if args.parallel:
            with multiprocessing.Pool(args.workers) as p:
                all_name = list(tqdm(p.imap(extractor.save, scenario_files), total=len(scenario_files)))
        else:
            for file in tqdm(scenario_files):
                extractor.save(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, default='/path/to/data_root')
    parser.add_argument("--save_dir", "-s", type=str, default="data/DeMo_processed",
                        help="output folder (datamodule.target.data_root of the training)")
    parser.add_argument("--workers", "-w", type=int, default=16, help="processes with --parallel")
    parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--num_historical_steps", type=int, default=50)
    parser.add_argument("--num_future_steps", type=int, default=60)
    parser.add_argument('--use_raceline', action='store_true')
    parser.add_argument('--use_raceline_velocity', action='store_true')
    parser.add_argument('--speed_cap_column', default=None,
                        help="parquet column with the speed cap in km/h (e.g. speed_cap_kph): stored at t0 in x_attr[:, 0] "
                             "for training/eval with use_speed_cap=True; default: no cap")
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help="splits to preprocess")

    args = parser.parse_args()
    preprocess(args)
