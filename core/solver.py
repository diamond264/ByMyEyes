from typing import List, Tuple, Dict

import os
import sys
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from core.llm import LLM
from core.logger import Logger
from core.token_utils import count_txt_tokens, count_img_tokens
from core.visualizer import Visualizer
from core.txt_generator import TextGenerator


INSTRUCTION = """### Instruction
You are an expert in sensor data analysis. \
Given the sensor data, determine the correct answer from the options listed in the question. \
Provide the answer with the format of <answer>ANSWER</answer>, \
where ANSWER corresponds to one of the options listed in the question. \
If the answer is not in the options, choose the most possible option."""

VIS_EXAMPLES_GUIDE = """Please refer to the examples provided in the images \
and use them to answer the following question for the target data."""

TXT_EXAMPLES_GUIDE = """Please refer to the provided examples \
and use them to answer the following question for the target data."""


class Solver:
    """Solver class for solving sensory tasks with LLMs"""

    def __init__(
        self,
        llm: LLM,
        config: Dict,
        task_metadata: Dict,
        logger: Logger,
    ):
        self.llm = llm
        self.config = config
        self.task_metadata = task_metadata
        self.logger = logger

    def get_ans_w_txt(
        self, data: np.array, examples: List[Tuple[np.array, str]], log_subdir: str
    ) -> str:
        """Get answer with text input"""
        tg = TextGenerator(
            self.task_metadata["channels"],
            self.task_metadata["sampling_rate"],
            self.config["txt_style"],
            self.config["txt_rounding_points"],
            self.config["txt_sampling_rate"],
        )
        tg_txt = tg.gen_txt(data)
        ex_txts = [tg.gen_txt(ex_data, ex_label) for ex_data, ex_label in examples]

        # compose txt
        txt_prompt = f"{INSTRUCTION}\n\n"
        txt_prompt += f"{self.task_metadata['data_description']} "
        txt_prompt += f"{TXT_EXAMPLES_GUIDE}\n\n"

        txt_prompt += "### Examples\n"
        for ex_txt in ex_txts:
            txt_prompt += f"{ex_txt}\n\n"

        txt_prompt += "### Question\n"
        txt_prompt += f"{tg_txt}\n"
        txt_prompt += f"*Question*: When the sensor data is used for {self.task_metadata['task_description'].strip('.')}, "
        txt_prompt += f"what is the most likely answer among {self.task_metadata['classes']}?\n*Answer*: "
        if self.config["use_cot"]:
            txt_prompt += "Let's think step-by-step. "

        prompt = [{"type": "text", "text": txt_prompt}]

        response = self.llm.generate(prompt)

        num_tokens = count_txt_tokens(txt_prompt, self.config["llm_version"])
        self.logger.store_chat(
            os.path.join(log_subdir, "task_solver.txt"), prompt, response, num_tokens
        )

        response = response.split("<answer>")[1].split("</answer>")[0].strip()
        if response.startswith("ANSWER: "):
            response = response.replace("ANSWER: ", "")
        return response

    def get_ans_w_vis(
        self, data: np.array, examples: List[Tuple[np.array, str]], log_subdir: str
    ) -> str:
        """Get answer with visualized input"""
        vs = Visualizer(
            self.task_metadata["channels"],
            self.task_metadata["sampling_rate"],
            self.config["vis_func"],
            self.config["vis_args"],
        )

        if self.config["vis_func"] == "raw waveform":
            ylim_max = data.max()
            ylim_min = data.min()
            for ex_data, _ in examples:
                ylim_max = max(ylim_max, ex_data.max())
                ylim_min = min(ylim_min, ex_data.min())
            self.config["vis_args"]["ylim"] = (ylim_min, ylim_max)
        # elif self.config["vis_func"] == "EMG muscle activation plot":
        #     self.config["vis_args"]["ylim"] = (0, 10)

        # compose imgs
        image_urls = []
        for i, (example_data, example_label) in enumerate(examples):
            ex_b64_img = vs.gen_b64_img(example_data, label=example_label)
            image_urls.append(ex_b64_img)
            example_label = example_label.replace(" ", "_")
            example_label = example_label.replace("/", "_")
            example_label = example_label.replace("-", "_")
            self.logger.store_img(
                os.path.join(log_subdir, f"{example_label}_{i}.png"), ex_b64_img
            )

        tg_b64_img = vs.gen_b64_img(data)
        image_urls.append(tg_b64_img)
        self.logger.store_img(os.path.join(log_subdir, "target.png"), tg_b64_img)
        vs.close()

        # compose txt
        txt_prompt = f"{INSTRUCTION}\n\n"
        txt_prompt += f"{self.task_metadata['data_description']} "
        txt_prompt += f"{VIS_EXAMPLES_GUIDE}\n\n"
        txt_prompt += "### Question\n"
        if self.config["use_knowledge"]:
            if self.config["vis_knowledge"] and self.config["vis_knowledge"] != "":
                txt_prompt += f"*Knowledge*: {self.config['vis_knowledge']}\n"
        txt_prompt += f"*Question*: When the sensor data is used for {self.task_metadata['task_description'].strip('.')}, "
        txt_prompt += f"what is the most likely answer among {self.task_metadata['classes']}?\n*Answer*: "
        if self.config["use_cot"]:
            txt_prompt += "Let's think step-by-step. "

        image_urls = [f"data:image/jpeg;base64,{url}" for url in image_urls]
        prompt = [
            {"type": "text", "text": txt_prompt},
            *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls],
        ]

        response = self.llm.generate(prompt)

        num_tokens = count_txt_tokens(txt_prompt, self.config["llm_version"])
        for url in image_urls:
            num_tokens += count_img_tokens(url)
        self.logger.store_chat(
            os.path.join(log_subdir, "task_solver.txt"), prompt, response, num_tokens
        )

        response = response.split("<answer>")[1].split("</answer>")[0].strip()
        if response.startswith("ANSWER: "):
            response = response.replace("ANSWER: ", "")
        return response

    def solve(
        self,
        data: np.array,
        examples: List[Tuple[np.array, str]],
        log_subdir: str = "",
    ) -> Tuple[str, str]:
        """Solve a sensory task with LLMs"""
        if self.config["use_vis"]:
            answer = self.get_ans_w_vis(data, examples, log_subdir)
        else:
            answer = self.get_ans_w_txt(data, examples, log_subdir)

        return answer
