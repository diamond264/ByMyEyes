import json
import yaml
import fire
import torch
import random
import datasets
import threading
import numpy as np

from typing import List, Tuple, Any, Union
from multiprocessing import Process, Manager
from sklearn.metrics import accuracy_score, f1_score

from core.logger import Logger
from core.llm import LLM
from core.vis_generator import VisualizationGenerator
from core.solver import Solver


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def gen_examples(ds: datasets.Dataset, num_examples: int) -> List[Tuple[np.array, str]]:
    examples = []
    random_idcs = np.random.choice(len(ds), num_examples, replace=False)
    for i in random_idcs:
        example_data = np.array(ds[int(i)]["data"])
        example_label = ds[int(i)]["label"]
        examples.append((example_data, example_label))

    return examples


def get_visualization(llm, task_metadata, examples, logger, results, lock):
    vg = VisualizationGenerator(llm, task_metadata, logger)
    vis_candidates = vg.plan("prompts/0_vis_plan")
    vis = vg.select(vis_candidates, examples, "prompts/0_vis_plan")

    with lock:
        results.append(vis)
        logger.print(f"visualization {vis['func']} selected")


def solve(
    solver: Solver,
    data: dict,
    ex_by_label: dict,
    config: dict,
    results: List[Any],
    lock: threading.Lock,
    pid: int,
) -> None:
    set_seed(config["seed"] + pid)

    examples = []
    for _, ex_ds in ex_by_label.items():
        examples += gen_examples(ex_ds, config["num_examples"])

    if len(examples) != config["num_examples"] * (len(ex_by_label)):
        examples = []
        for _, ex_ds in ex_by_label.items():
            examples += gen_examples(ex_ds, config["num_examples"])

    label_txt = "_".join(data["label"].split())
    label_txt = "_".join(label_txt.split("/"))
    label_txt = "_".join(label_txt.split("_"))
    log_dir = f"prompts/{pid}_{label_txt}"

    # plan visualization using LLM
    reset_vis_func = False
    if (config["use_vis"] and config["vis_func"] is None) or config["plan_vis"]:
        reset_vis_func = True
        vg = VisualizationGenerator(solver.llm, solver.task_metadata, solver.logger)
        vis_candidates = vg.plan(log_dir)
        vis = vg.select(vis_candidates, examples, log_dir)

        solver.logger.print(f"[{pid}] visualization {vis['func']} selected")
        config["vis_func"] = vis["func"]
        config["vis_args"] = vis["args"]
        config["vis_knowledge"] = vis["knowledge"]
        config["txt_style"] = vis["func"]
        config["txt_args"] = vis["args"]

    answer = solver.solve(np.array(data["data"]), examples, log_dir)

    if reset_vis_func:
        config["vis_func"] = None

    with lock:
        results.append((pid, data["label"], answer))
        solver.logger.print(f"[{pid}] GT: {data['label']}, Pred: {answer}")


def report(results: List[Any], logger: Logger) -> None:
    result_str = ""
    gts = []
    predictions = []
    for pid, gt, pred in results:
        result_str += f"[{pid}] GT: {gt}, Pred: {pred}\n"
        gts.append(gt)
        predictions.append(pred)

    logger.print(f"Accuracy: {accuracy_score(gts, predictions)}")
    logger.print(f"F1 Score: {f1_score(gts, predictions, average='macro')}")
    result_str += f"Accuracy: {accuracy_score(gts, predictions)}\n"
    result_str += f"F1 Score: {f1_score(gts, predictions, average='macro')}\n"

    logger.store("predictions.txt", result_str)


def run(config: str) -> None:
    with open(config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    set_seed(config["seed"])

    logger = Logger(config["log_dir"])
    logger.log_config(config)

    llm = LLM(
        model=config["llm_model"],
        version=config["llm_version"],
        llm_path=config["llm_path"],
    )
    logger.print("Loaded LLM")

    ds = datasets.load_from_disk(config["target_data_dir"])
    ds_by_label = {}
    for label in ds.unique("label"):
        ds_by_label[label] = ds.filter(lambda x, label=label: x["label"] == label)
        print(f"Label: {label}, Num: {len(ds_by_label[label])}")
    with open(config["task_metadata_path"], "r", encoding="utf-8") as f:
        task_metadata = json.load(f)
    logger.print("Loaded target data")

    solver = Solver(llm, config, task_metadata, logger)

    manager = Manager()
    results = manager.list()
    lock = manager.Lock()
    processes = []
    pid = 0

    # for each label in the target dataset filter samples
    tg_by_label = {}
    ex_by_label = {}
    for label in ds.unique("label"):
        len_ds = len(ds_by_label[label])
        tg_idcs = np.random.choice(len_ds, config["num_samples"], replace=False)
        tg_ds = ds_by_label[label].select(tg_idcs)
        tg_by_label[label] = tg_ds

        ex_idcs = [i for i in range(len_ds) if i not in tg_idcs]
        ex_ds = ds_by_label[label].select(ex_idcs)
        ex_by_label[label] = ex_ds

    logger.print("Solving tasks...")
    for label in ds.unique("label"):
        for data in tg_by_label[label]:
            pid += 1
            if config["multiprocessing"]:
                p = Process(
                    target=solve,
                    args=(solver, data, ex_by_label, config, results, lock, pid),
                )
                p.start()
                processes.append(p)

                if pid % config["num_process"] == 0:
                    for p in processes:
                        p.join()
                    processes = []
            else:
                solve(solver, data, ex_by_label, config, results, lock, pid)

    for p in processes:
        p.join()

    report(results, logger)


if __name__ == "__main__":
    fire.Fire(run)
