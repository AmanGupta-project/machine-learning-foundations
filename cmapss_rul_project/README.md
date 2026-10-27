C-MAPSS Turbofan Engine RUL Estimation

# Overview
This mini-project estimates the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset. The current notebook focuses on data preprocessing, feature extraction, and building a baseline predictive model (Linear Regression) to estimate engine failure.

# Dataset
The project uses the C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset provided by NASA. It contains simulated run-to-failure sensor data for aircraft engines.

# Project Structure
* `data/` : Folder containing the C-MAPSS dataset files (e.g., `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`).
* `Turbonfan_RUL_Estimate.ipynb` : Main Jupyter Notebook containing data exploration and model training.
* `requirements.txt` : Python dependencies needed to run this project.

## How to Run
1. Ensure Python is installed on your system.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt