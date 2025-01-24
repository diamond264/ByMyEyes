# By My Eyes: Grounding Multimodal Large Language Models with Sensor Data via Visual Prompting (EMNLP '24)
This is the official Python implementation of "By My Eyes: Grounding Multimodal Large Language Models with Sensor Data via Visual Prompting (EMNLP '24 main track, long paper)" by 
[Hyungjun Yoon](https://hjyoon.com/), 
Biniyam Aschalew Tolera, 
[Taesik Gong](https://taesikgong.com/), 
[Kimin Lee](https://sites.google.com/view/kiminlee), and 
[Sung-Ju Lee](https://sites.google.com/site/wewantsj/).

[![arXiv](https://img.shields.io/badge/arXiv-2407.10385-b31b1b.svg)](https://arxiv.org/abs/2407.10385) [![website](https://img.shields.io/badge/website-f9dc08.svg)](https://nmsl.kaist.ac.kr/projects/bymyeyes/) [![ACL](https://img.shields.io/badge/ACLAnthology-ed1c24.svg)](https://aclanthology.org/2024.emnlp-main.133/)

## Getting Started

### Installation

To set up the project, we recommend using Python 3.9.16 in a conda environment.
Follow the steps below to install and configure the environment.

1. Download and install [Anaconda](https://www.anaconda.com/).
2. Create and activate the conda environment using the commands below:

```bash
conda create -n bymyeyes python=3.9.16
conda activate bymyeyes
```

3. Once the environment is activated, install the required packages:

```bash
python -m pip install -r requirements.txt
```

### OpenAI API Key

To use the ChatGPT API, follow these steps to acquire your API key:

1. Visit [OpenAI](https://platform.openai.com/) and sign in.
2. Generate an API key in the **API** section.
3. Save the key to a file in your private directory.

```bash
echo "<your_openai_key>" > openai_key
```

### Datasets

To run the benchmarks, download the datasets from the following links:

- **[HHAR](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition)**
- **[UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)**
- **[Swim](https://github.com/brunnergino/swimming-recognition-lap-counting)**
- **[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)**
- **[Gesture](https://data.mendeley.com/datasets/ckwc76xr2z/2)**
- **[WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)**

Download and store the datasets in your storage directory.

### Preprocessing

We provide preprocessing scripts for the raw datasets. You can implement your own preprocessing with different settings. Ensure your data is in a compatible format with our framework by referring to the output structure from the provided script.

To run the preprocessing, use the following command:

```bash
python data_utils/preprocess.py --dataset <dataset_name> --data_dir <path_to_raw_data_directory> --out_dir <path_to_processed_data_directory>
```

## Run

### Configuration

We use YAML files for configuration. To set up your own configurations, refer to the `sample_config.yaml` file, which includes detailed comments to guide you in customizing the settings.

### Execution

After the configuratin, run the framework using the following command:

```bash
python run.py --config <config_file>
```

## Tested Environment

We tested our codes in this environment.

- OS: Ubuntu 20.04.4 LTS
- GPU: NVIDIA GeForce RTX 3090
- GPU Driver Version: 470.74
- CUDA Version: 11.2

## Citation

```
@inproceedings{yoon-etal-2024-eyes,
    title = "By My Eyes: Grounding Multimodal Large Language Models with Sensor Data via Visual Prompting",
    author = "Yoon, Hyungjun  and
      Tolera, Biniyam Aschalew  and
      Gong, Taesik  and
      Lee, Kimin  and
      Lee, Sung-Ju",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.133/",
    doi = "10.18653/v1/2024.emnlp-main.133",
    pages = "2219--2241",
    abstract = "Large language models (LLMs) have demonstrated exceptional abilities across various domains. However, utilizing LLMs for ubiquitous sensing applications remains challenging as existing text-prompt methods show significant performance degradation when handling long sensor data sequences. In this paper, we propose a visual prompting approach for sensor data using multimodal LLMs (MLLMs). Specifically, we design a visual prompt that directs MLLMs to utilize visualized sensor data alongside descriptions of the target sensory task. Additionally, we introduce a visualization generator that automates the creation of optimal visualizations tailored to a given sensory task, eliminating the need for prior task-specific knowledge. We evaluated our approach on nine sensory tasks involving four sensing modalities, achieving an average of 10{\%} higher accuracy compared to text-based prompts and reducing token costs by 15.8 times. Our findings highlight the effectiveness and cost-efficiency of using visual prompts with MLLMs for various sensory tasks. The source code is available at https://github.com/diamond264/ByMyEyes."
}
```
