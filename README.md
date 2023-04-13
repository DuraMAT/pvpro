# PV Production Tools (PV-Pro)

[![PyPI release](https://img.shields.io/pypi/v/pvpro.svg)](https://pypi.org/project/pvpro/)

In a typical photovoltaic (PV) system, more information is stored than just the DC or AC power. When a dataset contains the DC voltage, DC current, module temperature and plane-of-array irradiance, we can fit a single-diode model and extract many parameters as a function of time. These parameters include series resistance, shunt resistance, reference photocurrent, and more.

This package, PV-Pro, automates the analysis of PV production data to extract the rate of change of these parameters. 

**The package is still under active development. If there is any problem, please feel free to contact us!**

## Publications
Details of PV-Pro are provided in the following publications:

[1] Li, Baojie, Todd Karin, Bennet E. Meyers, Xin Chen, Dirk C. Jordan, Clifford W. Hansen, Bruce H. King, Michael G. Deceglie, and Anubhav Jain. "Determining circuit model parameters from operation data for PV system degradation analysis: PVPRO." Solar Energy 254 (2023): 168-181.

[2] Li, Baojie, Todd Karin, Xin Chen, and Anubhav Jain. "Estimation and Degradation Analysis of Physics-based Circuit Parameters for PV Systems Using Only DC Operation and Weather Data." In 2022 IEEE 49th Photovoltaics Specialists Conference (PVSC), pp. 1236-1236. IEEE, 2022.

# Methods
PV-Pro estimates essential PV module parameters using only operation (DC voltage and current) and weather data (irradiance and temperature). First, PV-Pro performs multi-stage data pre-processing to remove noisy data. Next, the time-series DC data are used to fit an equivalent circuit single-diode model (SDM) to estimate the circuit parameters by minimizing the differences between the measured and estimated values. In this way, the time evolutions of the SDM parameters are obtained.

![Image of PV-Pro methodology](https://github.com/DuraMAT/pvpro/blob/master/doc_img/pvpro_overview.png)

Here's a high level overview of the most important parts of the package.

- main.production_data_curve_fit - Fits a single diode model to production data.
- main.PvProHandler - class method for running the pvpro data analysis. Convenient way to keep track of all the variables required for the analysis and run production_data_curve_fit iteratively over time-series data.
- main.PvProHandler.execute - Runs the pvpro simulfit.

# Installation

## Install with pip
```
pip install pvpro
```



## Install with conda

Install can be performed with the included `pvpro-user.yml` file by running:

```
conda env create -f pvpro-user.yml
```
Next activate the environment, cd into the pvpro repository and run:

```
pip install -e .
```
## Make environment manually
Another way to make a valid virtual environment is with the following commands. This section will be updated in the future to make a more minimal environment.

```
conda create --name pvpro python=3 numpy scipy pandas matplotlib cvxpy tqdm pyqt
conda activate pvpro
conda install -c mosek mosek
pip install requests
pip install sklearn
pip install seaborn
pip install xlrd
pip install solar-data-tools statistical-clear-sky
pip install NREL-PySAM
pip install matplotlib==3.3.2
```

# Examples

## Run analysis on synthetic data

By generating a PV dataset with known module degradation, the performance of the algorithm in extracting single diode model parameters can be tested. A jupyter notebook showing the generation of dataset and analysis of this dataset is provided in [Synthetic_analyze.ipynb](examples/Synthetic_analyze.ipynb).  Estimated evolution trends of parameters show good match with the ground truth.

![Image of PV-Pro fit result of synthetic dataset](https://github.com/DuraMAT/pvpro/blob/master/doc_img/synthetic_results.png)

## Example analysis of real data.

The NIST ground array provides a useful testbed for PV-Pro [1]. A jupyter notebook showing analysis of this dataset is provided in [NIST_ground_array_analysis.ipynb](examples/NIST_ground_array_analysis.ipynb). 

PVPRO analysis fits a single diode model to the data at each timestep in the analysis. The trend of these parameters over time can be used to interpret what is degrading in the system. This analysis is only sensitive to module degradation (excepting drift in sensors) and not inverter degradation or downtime. Below, the PV-Pro results for this system show which parameters cause the observed power loss.

![Image of PV-Pro fit result](doc_img/nist_ground_result.png)

For this dataset, the estimated power degradation rate is -1.07%/yr. This system also appears to show an increase in series resistance over time. 





[1]. Boyd, M. (2017), Performance Data from the NIST Photovoltaic (PV) Arrays and Weather Station, Journal of Research (NIST JRES), National Institute of Standards and Technology, Gaithersburg, MD, [online], https://doi.org/10.6028/jres.122.040 (Accessed July 27, 2022)
