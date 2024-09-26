import os
import sys
import json

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from core.visualizer import Visualizer

VISUALIZATIONS = {
    "raw waveform": {
        "description": "This generates a raw signal of sensor data, displaying the amplitude of the signal over time. This is usually used to visualize the raw data and identify patterns in the signal.",
        "args": [],
    },
    "spectrogram": {
        "description": """This generates a spectrogram of sensor data, showing the density of frequencies over time. This is usually used to visualize the frequency components for high-frequency data which has features over components but is hard to figure out in the raw plot. It takes the length of the FFT used (nfft), the length of each segment (nperseg), and the number of points to overlap between segments (noverlap) as parameters. Different modes (mode) can be defined to specify the type of return values: ["psd" for power spectral density, "complex" for complex-valued STFT results, "magnitude" for absolute magnitude, "angle" for complex angle, and "phase" for unwrapped phase angle].""",
        "args": ["nfft", "nperseg", "noverlap", "mode"],
    },
    "signal power spectrum density": {
        "description": "This generates a power spectrum density plot, which shows the power of each frequency component of the signal on the x-axis. This is usually used to analyze the power distribution of different frequency components in the signal.",
        "args": [],
    },
    "EDA signal": {
        "description": "This generates a plot showing both raw and cleaned Electrodermal Activity (EDA) signals over time. This is usually used to analyze the EDA signals for patterns related to stress, arousal, or other psychological states.",
        "args": [],
    },
    "EDA skin conductance response (SCR)": {
        "description": "This generates a plot of skin conductance response (SCR) for EDA data, highlighting the phasic component, onsets, peaks, and half-recovery times. This is usually used to study the transient responses in EDA data related to specific stimuli or events.",
        "args": [],
    },
    "EDA skin conductance level (SCL)": {
        "description": "This generates a plot of skin conductance level (SCL) for EDA data over time. This is usually used to analyze the tonic component of EDA data, reflecting the overall level of arousal or stress over a period.",
        "args": [],
    },
    "ECG signal and peaks": {
        "description": "This generates a plot for Electrocardiogram (ECG) data, showing the raw signal, cleaned signal, and R peaks marked as dots to indicate heartbeats. This is usually used to analyze the heartbeats and detect anomalies in the ECG signal.",
        "args": [],
    },
    "ECG heart rate": {
        "description": "This generates a heart rate plot for ECG data, displaying the heart rate over time along with its mean value. This is usually used to monitor and analyze heart rate variability and trends over time.",
        "args": [],
    },
    "ECG individual heart beats": {
        "description": "This generates a plot of individual heartbeats and the average heart rate for ECG data. It aggregates heartbeats within an ECG recording and shows the average beat shape, marking P-waves, Q-waves, S-waves, and T-waves. This is usually used to study the morphology of individual heartbeats and identify irregularities.",
        "args": [],
    },
    # "RSP signal": {
    #     "description": "This generates a plot showing both raw and cleaned Respiration (RSP) signals over time, including exhalation and inhalation onsets and durations. This is usually used to analyze the breathing patterns and detect any abnormalities in respiration.",
    #     "args": [],
    # },
    # "RSP breathing rate": {
    #     "description": "This generates a breathing rate plot for RSP data, showing the breathing rate over time and its mean value. This is usually used to monitor and analyze the breathing rate and detect any irregularities.",
    #     "args": [],
    # },
    # "RSP breathing amplitude": {
    #     "description": "This generates a breathing amplitude plot for RSP data, displaying the amplitude of breaths over time. This is usually used to measure the depth of breathing and identify changes in breathing patterns.",
    #     "args": [],
    # },
    # "RSP respiratory volume per time": {
    #     "description": "This generates a plot of respiratory volume per time (RVT) for RSP data, showing RVT values and their mean over time. This is usually used to analyze the respiratory volume and detect any anomalies in breathing.",
    #     "args": [],
    # },
    # "RSP cycle symmetry": {
    #     "description": "This generates a cycle symmetry plot for RSP data, displaying peak-trough symmetry and rise-decay symmetry plots. This is usually used to study the symmetry of the breathing cycles and identify any asymmetries.",
    #     "args": [],
    # },
    "PPG signal and peaks": {
        "description": "This generates a plot for Photoplethysmogram (PPG) data, showing the raw signal, cleaned signal, and systolic peaks marked as dots. This is usually used to analyze the blood volume pulse and detect anomalies in the PPG signal.",
        "args": [],
    },
    "PPG heart rate": {
        "description": "This generates a heart rate plot for PPG data, displaying the heart rate over time and its mean value. This is usually used to monitor and analyze heart rate variability and trends over time based on PPG data.",
        "args": [],
    },
    "PPG individual heart beats": {
        "description": "This generates a plot of individual heartbeats and the average heart rate for PPG data, aggregating individual heartbeats within a PPG recording and showing the average beat shape. This is usually used to study the morphology of individual heartbeats based on PPG data.",
        "args": [],
    },
    "EMG signal": {
        "description": "This generates a plot showing both raw and cleaned Electromyogram (EMG) signals over time. This is usually used to analyze muscle activity and identify patterns in muscle contractions.",
        "args": [],
    },
    "EMG muscle activation": {
        "description": "This generates a muscle activation plot for EMG data, displaying the amplitudes of muscle activity and highlighting activated parts with lines. This is usually used to study muscle activation levels and identify specific periods of muscle activity.",
        "args": [],
    },
    "EOG signal": {
        "description": "This generates a plot showing both raw and cleaned Electrooculogram (EOG) signals over time, with blinks marked as dots. This is usually used to analyze eye movement patterns and detect blinks.",
        "args": [],
    },
    "EOG blink rate": {
        "description": "This generates a blink rate plot for EOG data, displaying the blink rate over time and its mean value. This is usually used to monitor and analyze the blink rate and detect any irregularities.",
        "args": [],
    },
    "EOG individual blinks": {
        "description": "This generates a plot of individual blinks for EOG data, aggregating individual blinks within an EOG recording and showing the median blink shape. This is usually used to study the morphology of individual blinks and identify patterns in blink dynamics.",
        "args": [],
    },
}

PLAN_INSTRUCTION = """### Instructions
You need to determine effective visualization methods for the given task. \
Provide visualization methods that aid in analyzing the data for this task, \
along with the required arguments for that method. \
Additionally, explain how to use the information from the visualization to solve the task. \
You can provide several candidates as a list. Generate the answer in the following format:
[{"func": visualization_method, "args": {"arg1": arg1_val, "arg2": arg2_val, ...}], "knowledge": knowledge}, ...]"""

SELECT_INSTRUCTION = """### Instruction
You do not have any prior knowledge about sensor data and visualization techniques.
Based solely on the visual cues in the provided images,
identify the visualization that most visually distinguishes \
all different classes for the given task.
Generate the answer in the following format:
{"func": visualization_method}"""

DEMONSTRATION = """### Demonstrations
Data description: The sensor data is collected from an accelerometer measuring acceleration along the x, y, and z axes. The data is normalized with the statistics of the user's data. The data is measured from an accelerometer attached to the ankles of a user.
Task description: A task for classifying running and walking activities using accelerometer data measured from an ankle-worn device.
Response: {"func": "raw waveform", "args": {}, "knowledge": "Use this to visualize the amplitude of the accelerometer signal over time. For classifying running and walking, observe the patterns in the waveform: running typically shows higher amplitude and more frequent peaks due to the higher impact and faster motion, while walking shows lower amplitude and less frequent peaks."}

Data description: The sensor data is collected from an accelerometer measuring acceleration along the x, y, and z axes. The data is normalized with the statistics of the vehicle's data. The data is measured from an accelerometer attached to a vehicle.
Task description: A task for classifying road types, such as asphalt, dirt, and cobblestone, using accelerometer data measured from a vehicle.
Response: {"func": "spectrogram", "args": {"nfft": 128, "nperseg": 128, "noverlap": 120, "mode": "magnitude"], "knowledge": "Use this to analyze the frequency components of the accelerometer signal over time. The colors in the spectrogram represent the magnitude of the frequencies: brighter colors indicate higher magnitudes. For road type classification, asphalt typically shows lower frequency components with smoother patterns, dirt shows higher frequency components with irregular patterns, and cobblestone shows high-frequency components with periodic patterns due to the regular bumps."}

Data description: The sensor data is collected from an ECG measuring the electrical activity of the heart. The data is normalized with the statistics of the user's data. The data is measured using electrodes attached to the chest of a user.
Task description: A task for detecting sleep apnea using ECG data measured from chest electrodes.
Response: {"func": "ECG individual heart beats", "args": {}, "knowledge": "Use this to aggregate and visualize individual heartbeats within an ECG recording. In normal beats, the P-wave precedes the QRS complex, and the T-wave follows it. In sleep apnea, irregularities in the intervals between the P, Q, R, S, and T peaks can be observed. For instance, the absence of regular QRS complexes or prolonged intervals can indicate episodes of apnea. The plot helps identify these patterns by showing the average shape of the heartbeats and marking the specific peaks."}

Data description: The sensor data is collected from an EMG sensor measuring muscle electrical activity. The data is normalized with the statistics of the user's data. The data is measured using electrodes attached to the forearm of a user.
Task description: A task for recognizing finger gestures, such as numbers, using EMG data measured from forearm electrodes.
Response: {"func": "EMG signal", "args": {}, "knowledge": "Use this to visualize the raw EMG signal over time. For recognizing finger gestures, observe the patterns and amplitude of muscle activity. Different numbers (gestures) will produce distinct patterns in the EMG signal. For example, bending more fingers usually results in higher amplitude signals due to increased muscle activation."}

Data description: The sensor data is collected from an ECG measuring the electrical activity of the heart. The data is normalized with the statistics of the user's data. The data is measured using electrodes attached to the chest of a user.
Task description: A task for detecting whether the user is running or not using ECG data measured from chest electrodes.
Response: {"func": "ECG heart rate", "args": {}, "knowledge": "Use this to monitor heart rate over time and analyze activity levels. A significant increase in heart rate can indicate that the user is running. The plot should show a higher average heart rate during running periods compared to resting or walking periods. Sudden spikes and consistent high heart rates are typical indicators of running."}"""


class VisualizationGenerator:
    def __init__(self, llm, task_metadata, logger):
        self.visualizations = VISUALIZATIONS
        self.llm = llm
        self.logger = logger
        self.task_desc = task_metadata["task_description"]
        self.data_desc = task_metadata["data_description"]
        self.sr = task_metadata["sampling_rate"]
        self.channels = task_metadata["channels"]

    def vis_descriptions(self):
        description_text = ""
        for vis, description in self.visualizations.items():
            description_text += f"*{vis}*: {description['description']}"
            if len(description["args"]) > 1:
                description_text += f" (Arguments: {', '.join(description['args'])})"
            description_text += "\n"
        description_text = description_text.strip()

        return description_text

    def get_planning_prompt(self):
        prompt = f"{PLAN_INSTRUCTION}\n\n"
        prompt += "The available visualization methods are as follows:\n\n"
        prompt += f"{self.vis_descriptions()}\n\n"
        prompt += f"{DEMONSTRATION}\n\n"
        prompt += "### Question\n"
        prompt += f"Task description: {self.task_desc.capitalize()}\n"
        prompt += f"Data description: {self.data_desc.capitalize()}\n"
        prompt += "Response: " ""
        prompt = [{"type": "text", "text": prompt}]

        return prompt

    def get_selection_prompt(self, candidates, examples, log_dir):
        img_urls = []
        for candidate in candidates:
            if candidate["func"] == "raw waveform":
                ylim_max = -float("inf")
                ylim_min = float("inf")
                for ex_data, _ in examples:
                    ylim_max = max(ylim_max, ex_data.max())
                    ylim_min = min(ylim_min, ex_data.min())
                candidate["args"]["ylim"] = (ylim_min, ylim_max)

            visualizer = Visualizer(
                self.channels, self.sr, candidate["func"], candidate["args"]
            )
            for i, (example_data, example_label) in enumerate(examples):
                b64_img = visualizer.gen_b64_img(example_data, example_label)
                img_urls.append(b64_img)
                example_label = example_label.replace(" ", "_")
                example_label = example_label.replace("/", "_")
                example_label = example_label.replace("-", "_")
                self.logger.store_img(
                    os.path.join(
                        log_dir, f"{candidate['func']}_{example_label}_{i}.png"
                    ),
                    b64_img,
                )
            visualizer.close()

        txt_prompt = f"{SELECT_INSTRUCTION}\n\n"
        txt_prompt += "### Question\n"
        txt_prompt += f"Visualization methods: {[candidate['func'] for candidate in candidates]}\n"
        txt_prompt += f"Task description: {self.task_desc.capitalize()}\n"
        txt_prompt += f"Data description: {self.data_desc.capitalize()}\n"
        txt_prompt += "Response: "

        urls = [f"data:image/jpeg;base64,{url}" for url in img_urls]
        prompt = [
            {"type": "text", "text": txt_prompt},
            *[{"type": "image_url", "image_url": {"url": f"{url}"}} for url in urls],
        ]

        return prompt

    def select(self, candidates, examples, log_dir):
        prompt = self.get_selection_prompt(candidates, examples, log_dir)
        res = self.llm.generate(prompt)
        self.logger.store_chat(os.path.join(log_dir, "vis_selection.txt"), prompt, res)

        while res[0] != "{":
            res = res[1:]
        while res[-1] != "}":
            res = res[:-1]
        res = json.loads(res)

        for candidate in candidates:
            if candidate["func"] == res["func"]:
                return candidate

    def plan(self, log_dir):
        prompt = self.get_planning_prompt()
        res = self.llm.generate(prompt)
        self.logger.store_chat(os.path.join(log_dir, "vis_plan.txt"), prompt, res)
        while res[0] != "[":
            res = res[1:]
        while res[-1] != "]":
            res = res[:-1]
        res = json.loads(res)

        return res
