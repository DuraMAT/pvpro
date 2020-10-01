# PV Production Tools (pvpro)

The typical photovoltaic (PV) analysis tools focus on extracting the rate of change of power at reference conditions. This quantity, *pmp_ref*, is just one of many physical parameters of a PV system. 

Fortunately, in a typical PV dataset, more information is stored than just the DC or AC power. When a dataset also contains the DC voltage, DC current, module temperature and plane-of-array irradiance, we can fit a single-diode model and extract many parameters as a function of time. These parameters include series resistance, shunt resistance, reference photocurrent, and more.

This package, pvpro, automates the analysis of PV production data to extract the rate of change of these parameters. 

**The package is still under active development so don't expect it to work perfectly yet!**

To try it out, you need to clone the most recent development branch of solar-data-tools and statistical-clear-sky.

# Methods

Here's a high level overview of the most important parts of the package.

- fit.production_data_curve_fit - Fits a single diode model to production data.
- main.PvProHandler - class method for running the pvpro data analysis. Convenient way to keep track of all the variables required for the analysis and run production_data_curve_fit iteratively over time-series data.
- main.PvProHandler.execute - Runs the pvpro simulfit.


# Installation

## Install with conda

Install can be performed with the included `pvpro-user.yml` file by running:

```
conda env create -f pvpro-user.yml
```
Next activate the environment, cd into the pvpro repository and run:

```
pip install -e .
```

## Install with pip
```
pip install pvpro
```

## Make environment by installing things seperately

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