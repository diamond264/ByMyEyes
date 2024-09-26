# By My Eyes: Grounding Multimodal Large Language Models with Sensor Data via Visual Prompting

 #### Datasets
Download the datastes from the following link, preprocessing code is provided in this r
| Dataset       | Link    |
| ------------- | ------- |
| HHAR          | [HHAR link](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition)  |
| UTD-MHAD      | [UTD-MHAD link](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)  |
| Swim          | [Swim link](https://github.com/brunnergino/swimming-recognition-lap-counting)  |
| PTB-XL        | [PTB-XL link](https://physionet.org/content/ptb-xl/1.0.3/)  |
| Gesture       | [Gesture link](https://data.mendeley.com/datasets/ckwc76xr2z/2)  |
| WESAD         | [WESAD link](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)  |

#### Instructions

After cloning the repository, install and set up the environment with `requirements.txt`

```bash
# Create a conda environment
conda create -n by-my-eyes python==3.9.16

# Activate the conda environment
conda activate by-my-eyes

# Install dependencies
pip install -r requirements.txt

# Running inference
python run.py --config sample_config.yaml
```


