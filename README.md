# Trustworthy Digital-Twin-Based Deep Reinforcement Learning for PV–Battery Energy Cyber-Physical Systems

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Modelica](https://img.shields.io/badge/Modelica-FMU-orange.svg)](https://modelica.org/)
[![FMI](https://img.shields.io/badge/FMI-2.0-green.svg)](https://fmi-standard.org/)
[![RL](https://img.shields.io/badge/DRL-PPO-purple.svg)](https://stable-baselines3.readthedocs.io/)
[![Status](https://img.shields.io/badge/status-master's%20thesis%20prototype-yellow.svg)](#project-status)

## Overview

This repository contains the research implementation for the Master's thesis:

> **Trustworthy Digital-Twin-Based Deep Reinforcement Learning for PV–Battery Energy Cyber-Physical Systems**

The project develops and evaluates an end-to-end energy-management framework for a grid-connected PV–battery microgrid. A Modelica physical model is exported as an FMI 2.0 Co-Simulation FMU and connected to Python through FMPy. The FMU is wrapped as a Gymnasium environment and controlled using Stable-Baselines3 Proximal Policy Optimization (PPO).

The framework combines:

- a Modelica/FMI digital twin of the PV–battery system;
- 15-minute Python–FMU co-simulation;
- LSTM forecasting of solar irradiance and electrical demand;
- MDP-PPO and forecast-augmented POMDP-LSTM-PPO controllers;
- a semi-formal runtime safety shield;
- rule-based controller baselines;
- physical, environmental, economic, safety, timing, and XAI evaluation.

The main contribution is the **integration and evaluation** of forecasting, learned control, runtime constraint enforcement, physical digital-twin feedback, and explainability. The work does not claim that PPO, LSTM, or runtime shielding are individually new.

---

## System Architecture

```mermaid
flowchart LR
    A[Weather, Load, Price and CO2 Data] --> B[LSTM Forecasting]
    A --> C[Gymnasium Environment]
    B --> C
    C --> D[PPO Controller]
    D --> E[Runtime Safety Shield]
    E --> F[Modelica FMI 2.0 FMU]
    F --> G[Physical Outputs]
    G --> C
    D --> H[XAI and Policy Analysis]
    E --> H
    G --> I[Performance, Safety and Economic Evaluation]
```

At every 15-minute control step:

1. The environment reads weather, demand, electricity-price, and carbon-intensity data.
2. The controller observes the current system state, with or without future GHI/load forecasts.
3. PPO proposes a continuous battery command.
4. The safety layer accepts, clips, blocks, or replaces the proposed command when necessary.
5. The FMU executes the command and returns the next physical state.
6. Reward, safety diagnostics, timing data, and explanation logs are recorded.

---

## Controllers

| Controller | Observation | Forecast information | Runtime enforcement |
|---|---:|---|---|
| Simple RBC | Rule based | No | Rule constraints |
| Enhanced RBC | Rule based | No | Improved deterministic rules |
| MDP-PPO | 12 variables | No future GHI/load | Basic environment protection |
| POMDP-LSTM-PPO | 20 variables | GHI and load at +1 h to +4 h | Basic environment protection |
| Shielded POMDP-LSTM-PPO | 20 variables | GHI and load at +1 h to +4 h | SOC, voltage, ramp, reserve, clipping, blocking, and fallback |

### Important terminology

The PPO policy is **not recurrent**. The LSTM is an external forecasting model.  
`POMDP-LSTM-PPO` therefore means a feedforward PPO policy whose observation is augmented with LSTM forecasts.

---

## Control and Training Configuration

| Item | Value |
|---|---|
| Control interval | 900 s = 15 min |
| Battery action | Continuous, approximately `[-50, +50] kW` |
| Negative action | Battery charging |
| Positive action | Battery discharging |
| PPO policy | `MlpPolicy` |
| Training timesteps | 500,000 |
| Training period | Q1–Q3 2023 |
| Final holdout | Q4 2023 |
| Learning rate | 0.0003 |
| Rollout length | 2,048 |
| Batch size | 64 |
| Epochs per rollout | 10 |
| Discount factor | 0.999 |
| GAE lambda | 0.95 |
| PPO clip range | 0.2 |
| Entropy coefficient | 0.0 |
| Value-function coefficient | 0.5 |
| Maximum gradient norm | 0.5 |
| Main seed | 42 |

The internal evaluation window is used for checkpoint selection inside the Q1–Q3 development period. It is not treated as an independent unseen test set. Q4 remains the late-year holdout.

---

## Runtime Safety Shield

The safety supervisor is placed between PPO and the physical FMU. It uses explicit operational rules to calculate a safe action interval before execution.

Main constraints include:

- hard SOC envelope: `0.10–0.90`;
- reporting SOC limits: `0.15–0.85`;
- operational guard band: `0.17–0.83`;
- forecast-aware lower SOC reserve;
- PCC voltage range: `0.95–1.05 p.u.`;
- battery power limits;
- ramp-rate limit;
- predicted next-step SOC and voltage checks;
- conservative fallback action when required.

The implementation is best described as a **semi-formal runtime shield** or a **formalized-constraint safety layer**. It is not a universally verified controller based on theorem proving, reachability analysis, or control barrier functions.

---

## Repository Contents

Recommended public repository structure:

```text
trustworthy-ecps-drl/
├── README.md
├── notebooks/
│   └── Untitled22 2.ipynb
├── data/
│   └── master_environment_2023.csv
├── fmu/
│   └── ECPS_DigitalTwin.fmu
├── docs/
│   └── technical_brief.pdf
├── outputs/                    # Generated results;
├──                 
└── LICENSE                     
```

Suggested renaming of the supplied files:



The supplied CSV contains 35,041 time-indexed rows at 15-minute resolution and includes load, temperature, GHI, electricity price, carbon intensity, calendar features, and +1 h to +4 h future-price features.

The FMU was generated with OpenModelica v1.26.1 and exposes inputs for battery command, irradiance, temperature, and load, together with outputs including SOC, PV power, load, grid power, PCC voltage, battery throughput, loss proxy, residual, and stability flag.

---

## Requirements

Primary Python dependencies:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
gymnasium
stable-baselines3[extra]
fmpy
```

A suitable installation command is:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn torch \
    gymnasium "stable-baselines3[extra]" fmpy
```

The current notebook was developed for **Google Colab** and contains Google Drive paths. Local execution requires changing those paths.

---

## Running the Project in Google Colab

1. Create the recommended repository folders.
2. Upload the notebook, dataset, FMU, and technical brief.
3. Open `notebooks/Trustworthy_ECPS_Pipeline.ipynb` in Google Colab.
4. Add the data and FMU files to Google Drive.
5. Update the path variables in the first notebook cell:

```python
BASE_DIR = "/content/drive/MyDrive/micro grid data"
FMU_PATH = "/content/drive/MyDrive/ECPS_DigitalTwin.fmu"
ORIGINAL_CSV = f"{BASE_DIR}/master_environment_2023.csv"
```

6. Run the cells from top to bottom.

The notebook performs the following stages:

1. dependency setup and reproducibility configuration;
2. FMU loading and low-level co-simulation wrapper creation;
3. future-price feature preparation;
4. LSTM training and Q4 forecast evaluation;
5. Gymnasium environment construction;
6. MDP-PPO training;
7. POMDP-LSTM-PPO training;
8. safety-shield construction and shielded PPO training;
9. short-window and complete-period controller evaluation;
10. techno-economic, trustworthiness, overhead, and XAI analysis.

Training three PPO controllers for 500,000 timesteps can require substantial runtime. A GPU can accelerate the LSTM stage, although FMU interaction remains mainly CPU based.

---

## Running Locally

For local execution, remove or replace the Colab-specific code:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then use paths relative to the repository root, for example:

```python
from pathlib import Path

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "master_environment_2023.csv"
FMU_PATH = ROOT / "fmu" / "ECPS_DigitalTwin.fmu"
OUTPUT_DIR = ROOT / "outputs"
```

Run the notebook with:

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

---

## Main Reported Results

The following results correspond to selected checkpoints and the assumptions documented in the technical brief.

| Evaluation | Completion | CO2 | Self-sufficiency | Self-consumption | Grid import |
|---|---:|---:|---:|---:|---:|
| Shielded POMDP, Q4 | 100% | 1.0476 t | 56.05% | 84.74% | 3,184 kWh |
| Shielded POMDP, full year | 100% | 2.8061 t | 71.28% | 68.34% | 9,068 kWh |

Additional findings:

- LSTM forecasts outperformed persistence for GHI and load from +1 h to +4 h.
- Forecast augmentation reduced risky-action frequency but did not guarantee safe operation.
- The unshielded forecast-aware controller failed near both lower and upper SOC boundaries.
- The shield restored complete Q4 and full-year operation in the evaluated scenarios.
- Mean shield decision time was approximately `0.37 ms`.
- Mean shielded environment-transition time was approximately `1.01 ms`.
- Improved local XAI achieved mean local surrogate \(R^2\) values from approximately `0.950` to `0.974`.
- The shielded controller improved emissions, autonomy, grid import, voltage behavior, and composite economic value, but increased battery activity and the heat/loss proxy.

The reported economic result is a **composite techno-economic value**, not a direct electricity bill. It depends on the stated carbon, battery-wear, failure, and local-resilience assumptions.

---

## Generated Outputs

Depending on which cells are executed, the notebook generates:

- LSTM forecast metrics and comparison plots;
- PPO Monitor logs;
- TensorBoard logs;
- checkpoint models;
- deterministic evaluation files;
- training reward and episode-length plots;
- SOC protection and shield-intervention summaries;
- three-day behavioral comparisons;
- Q4 and full-year performance tables;
- computational-overhead summaries;
- trustworthiness and safety metrics;
- XAI feature-importance and local-fidelity results.

Trained PPO model checkpoints are generated by the notebook and are not necessarily included in the repository unless uploaded separately.

---

## Reproducibility

The main experiments use seed `42` for Python, NumPy, PyTorch, and Stable-Baselines3.

For stronger statistical evidence, future releases should add:

- multiple PPO seeds;
- exact package versions in `requirements.txt`;
- saved best-evaluation checkpoints;
- machine and operating-system information;
- multi-year and multi-climate datasets;
- sensitivity analysis for safety thresholds and economic coefficients.

---

## Limitations

- The current headline PPO results are primarily based on one random seed.
- The study covers one year and one climate/data setting.
- The runtime shield is semi-formal rather than universally verified.
- Battery degradation is represented mainly through throughput and heat/loss proxies.
- The timing study does not fully include sensor, communication, actuator, or deployment-hardware delays.
- The notebook contains Colab-specific paths and should be modularized for production use.
- The FMU exposes a global `stability_flag`; more detailed Modelica diagnostic flags would improve root-cause analysis.
- The code is a research prototype and should not be used to control a physical battery without independent validation, hardware protection, and engineering approval.

---

## Project Status

This repository is a Master's thesis research prototype. It is intended for academic reproducibility, method demonstration, and continued research on trustworthy AI for energy cyber-physical systems.

---

## Citation

When using this repository, please cite:

```bibtex
@mastersthesis{seyyedi2026trustworthy,
  author  = {Seyyedi, Mahdi},
  title   = {Trustworthy Digital-Twin-Based Deep Reinforcement Learning for PV--Battery Energy Cyber-Physical Systems},
  school  = {University of Genoa},
  year    = {2026},
  type    = {Master's thesis}
}
```

Update the final bibliographic information after thesis submission or publication.

---

## Author

**Mahdi Seyyedi**  
Master's thesis project, University of Genoa

---

## License

No license has yet been selected.

Before making the repository public, verify that the dataset, FMU, figures, and third-party materials may be redistributed. Add an appropriate open-source license only after confirming those rights. Without a license, other users may view the repository but do not automatically receive permission to reuse or modify the code.
