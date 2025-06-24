<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

##Bachelor Thesis on the topic of Latency Language-Action Foundation Models via Upside-Down RL. 

## Project Organization

```
├── docs                       <- File documentation in HTML using Sphinx
├── jobscripts                 <- Shell files used to run jobstricts on a HPC
├── models                     <- Trained and serialized models
├── notebooks                  <- Jupyter notebooks used in the research, mostly for data processing and initial testing
├── plots                      <- Plots generated and used in the project
├── reports                    <- The thesis related to the project
├── running-outputs            <- Outputs obtained when running some of the more important parts of the code, named accordingly
├── LICENSE                    <- Open-source license
├── pyproject.toml             <- Project configuration file with package metadata
├── uv.lock                    <- The requirements of the project, run via UV package manager 
├── setup.cfg                  <- Configuration file for flake8
└── lukasz_sawala_bsc_thesis   <- Source code for use in this project
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

## Acknowledgements

Special thanks to Habrok, the HPC cluster of the University of Groningen, for making this thesis possible by granting unlimited access to its resources.
--------

