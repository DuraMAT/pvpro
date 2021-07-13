# PV Production Tools (pvpro)

[![PyPI release](https://img.shields.io/pypi/v/pvpro.svg)](https://pypi.org/project/pvpro/)

The typical photovoltaic (PV) analysis tools focus on extracting the rate of change of power at reference conditions. This quantity, *pmp_ref*, is just one of many physical parameters of a PV system. 

Fortunately, in a typical PV dataset, more information is stored than just the DC or AC power. When a dataset also contains the DC voltage, DC current, module temperature and plane-of-array irradiance, we can fit a single-diode model and extract many parameters as a function of time. These parameters include series resistance, shunt resistance, reference photocurrent, and more.

This package, pvpro, automates the analysis of PV production data to extract the rate of change of these parameters. 

**The package is still under active development so don't expect it to work perfectly yet!**

# Methods

Here's a high level overview of the most important parts of the package.

- fit.production_data_curve_fit - Fits a single diode model to production data.
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

## Estimate size of power block
The size of a power block can be estimated by first estimating `vmp_ref` and `imp_ref`. The number of modules in series in and parallel are then found by dividing by the datasheet  

An example with the NIST ground dataset is provided in the file [example_estimate_number_series_parallel.py](examples/example_estimate_number_series_parallel.py)

## Run analysis on synthetic data

By generating a PV dataset with known module degradation, the performance of the algorithm in extracting single diode model parameters can be tested.

First, generate sythetic data using [synth01_generate_synthetic_data.py](examples/synth01_generate_synthetic_data.py). Next, run the time series parameter extraction in [synth02_simulfit.py](examples/synth02_simulfit.py).

## Example analysis of real data.

The NIST ground array provides a useful testbed for PVPRO [1]. A jupyter notebook showing analysis of this dataset is provided in [NIST16_analyze.ipynb](examples/NIST16_analyze.ipynb). 

PVPRO analysis fits a single diode model to the data at each timestep in the analysis. Below, the thoery lines are shown next to the cleaned data.

![Image of PVPRO fit result](https://github.com/DuraMAT/pvpro/blob/master/doc/img/nist-ground-fit-result.png)

The trend of these parameters over time can be used to interpret what is degrading in the system. This analysis is only sensitive to module degradation (excepting drift in sensors) and not inverter degradation or downtime. Below, the PVPRO results for this system show which parameters cause the observed power loss.

![Image of PVPRO parameter trend](https://github.com/DuraMAT/pvpro/blob/master/doc/img/nist-ground-parameter-trend.png)

For this dataset, the estimated power degradation rate is -1.7%/yr. Inspecting the first panel, photocurrent loss is estimated to cause a -0.8%/yr loss in power, making photocurrent loss responsible for 47% of the observed power loss. This system also appears to show an increase in series resistance over time. 





[1]. Boyd, M. (2017), Performance Data from the NIST Photovoltaic (PV) Arrays and Weather Station, Journal of Research (NIST JRES), National Institute of Standards and Technology, Gaithersburg, MD, [online], https://doi.org/10.6028/jres.122.040 (Accessed July 13, 2021)
