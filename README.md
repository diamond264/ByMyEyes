# By My Eyes: Grounding Multimodal Large Language Models with Sensor Data via Visual Prompting (EMNLP '24)
This is the official Python implementation of "By My Eyes: Grounding Multimodal Large Language Models with Sensor Data via Visual Prompting (EMNLP '24 main track, long paper)" by 
[Hyungjun Yoon](https://hjyoon.com/), 
Biniyam Aschalew Tolera, 
[Taesik Gong](https://taesikgong.com/), 
[Kimin Lee](https://sites.google.com/view/kiminlee), and 
[Sung-Ju Lee](https://sites.google.com/site/wewantsj/).

[![arXiv](https://img.shields.io/badge/arXiv-2407.10385-b31b1b.svg)](https://arxiv.org/abs/2407.10385)

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
@article{yoon2024my,
  title={By My Eyes: Grounding Multimodal Large Language Models with Sensor Data via Visual Prompting},
  author={Yoon, Hyungjun and Tolera, Biniyam Aschalew and Gong, Taesik and Lee, Kimin and Lee, Sung-Ju},
  journal={arXiv preprint arXiv:2407.10385},
  year={2024}
}
```
