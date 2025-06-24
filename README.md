<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Bachelor Thesis on the topic of Low-Latency Language-Action Foundation Models via Upside-Down RL. 

## 🌲 Project Organization

```
├── docs                       <- File documentation in HTML using Sphinx
├── jobscripts                 <- Shell files used to run jobscripts on an HPC
└── lukasz_sawala_bsc_thesis   <- Source code for use in this project
│   └── .FILE_EXPLANATIONS     <- Brief comments about every file made for clarity
├── models                     <- Trained and serialized models
├── notebooks                  <- Jupyter notebooks used in the research, mostly for data processing and initial testing
├── plots                      <- Plots generated and used in the project
├── reports                    <- The thesis related to the project
├── running-outputs            <- Outputs obtained when running some of the more important parts of the code, named accordingly
├── LICENSE                    <- Open-source license
├── pyproject.toml             <- Project configuration file with package metadata
├── uv.lock                    <- The requirements of the project, run via UV package manager 
├── setup.cfg                  <- Configuration file for flake8
```
---

## 🏃‍♂️ Running Source Code
### 🛠️ Set-Up

**Clone the Repository**: 
Start by cloning the repository to your local machine.
   ```bash
   git clone https://github.com/LukaszSawala/Upside-Down-RL-Transformer.git
   cd Upside-Down-RL-Transformer
   ```
**Set Up Package Environment**:
    Download uv package manager by running the following command:
    
   ```bash
   pip install uv
   ```
    
   Make sure all dependencies are installed by running the following command:
   ```bash
   uv sync
   ```

---
### **⚙️ Parser Arguments**

| Argument               | Type    | Default      | Description                                                                                          |
|------------------------|---------|---------------|------------------------------------------------------------------------------------------------------|
| `--seed`               | `int`   | `0`           | Seed for random number generation to ensure reproducibility.                                         |
| `--model_type`         | `str`   | `"NeuralNet"` | Model architecture to use. Choices: `NeuralNet`, `DecisionTransformer`, `BERT_UDRL`, `BERT_MLP`, `ANTMAZE_BERT_MLP`, `ANTMAZE_NN`. |
| `--episodes`           | `int`   | `15`          | Number of evaluation episodes to run.                                                                |
| `--d_r_array_length`   | `int`   | `36`          | *(Evaluation only)* Length of the desired return array passed to the model.                          |
| `--epochs`             | `int`   | `15`          | *(Training only)* Number of epochs to train the model.                                               |
| `--hidden_size`        | `int`   | `256`         | *(Training only)* Hidden layer size in the model.                                                    |
| `--batch_size`         | `int`   | `32`          | *(Training only)* Batch size used during training.                                                   |
| `--learning_rate`      | `float` | `3e-4`        | *(Training only)* Learning rate for the optimizer.                                                   |
| `--patience`           | `int`   | `2`           | *(Training only)* Early stopping patience threshold.                                                 |

> Note: Arguments marked *(Training only)* or *(Evaluation only)* are conditionally included depending on the execution mode.

---

### ✅ Example usage

```bash
# Activate virtual environment and run evaluation
source .venv/bin/activate
cd lukasz_sawala_bsc_thesis/
python transfer_eval_various_conditions.py --episodes 10 --model_type ANTMAZE_BERT_MLP --d_r_array_length 21
```
---

## Acknowledgements

Special thanks to Habrok, the HPC cluster of the University of Groningen, for making this thesis possible by granting unlimited access to its resources.
--------

