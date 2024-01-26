# PV Production Tools (PV-Pro)

[![PyPI release](https://img.shields.io/pypi/v/pvpro.svg)](https://pypi.org/project/pvpro/)

<img src="https://github.com/DuraMAT/pvpro/blob/master/doc_img/pvpro_logo_slogan.gif?raw=true" width="800"/>
<img src="https://github.com/DuraMAT/pvpro/blob/master/doc_img/pvpro_overview_new.png?raw=true" width="800"/>

In a typical photovoltaic (PV) system, more information is stored than just the DC or AC power. When a dataset contains the **Production** (DC voltage & DC current) and **Weather data** (module temperature and plane-of-array irradiance) , we can fit and reconstruct a **precise physical model** of the PV system. This model serves to:

 - **Identify the degradation trend and rate** of key PV parameters
 - **Perform irradiance-to-power conversion** for accurate power prediction

**The package is still under active development. If there is any problem, please feel free to [contact us](mailto:baojieli@lbl.gov)!**

## Publications
Details of PV-Pro are provided in the following publications. If you use PV-Pro in a published work, please cite:

[1] Li, B., et al. "Determining circuit model parameters from operation data for PV system degradation analysis: PVPRO." Solar Energy 254 (2023): 168-181. DOI: [10.1016/j.solener.2023.03.011](https://doi.org/10.1016/j.solener.2023.03.011)

[2] Li, B., et al. "Detection and Analyze of Off-Maximum Power Points of PV Systems Based on PV-Pro Modelling." In 2023 IEEE 50th Photovoltaic Specialists Conference (PVSC), pp. 1-3. IEEE, 2023. DOI: [10.1109/PVSC48320.2023.10359868](https://doi.org/10.1109/PVSC48320.2023.10359868)

[3] Li, B., et al. "Estimation and Degradation Analysis of Physics-based Circuit Parameters for PV Systems Using Only DC Operation and Weather Data." In 2022 IEEE 49th Photovoltaics Specialists Conference (PVSC), pp. 1236-1236. IEEE, 2022. DOI: [10.1109/PVSC48317.2022.9938484](https://doi.org/10.1109/PVSC48317.2022.9938484)

# Installation

```
pip install pvpro==0.1.3
```

### Install Mosek solver (Optional)

Pre-processing of PV-Pro could use [solar-data-tools](https://github.com/slacgismo/solar-data-tools) for better performance (optional), which requires the installation of [Mosek](https://www.mosek.com/resources/getting-started/) solver. MOSEK is a commercial software package. You will still need to obtain a license. More information is available here:

* [Free 30-day trial](https://www.mosek.com/products/trial/)
* [Personal academic license](https://www.mosek.com/products/academic-licenses/)


# Methodology

## Method
PV-Pro can estimates 10 essential PV module parameters (listed below) at the reference condition (STC) using only production (DC voltage and current) and weather data (irradiance and temperature). Specifically, PV-Pro has 2 steps:
- **Pre-processing**: Identify outliers, clear sky, operating conditions, etc.
- **Parameter extraction**: Fit a single-diode model (SDM) to get the estimated SDM parameters by minimizing the differences between the measured and modeled voltage & current. Then use the SDM parameters to estimate the IV parameters at STC.

| SDM parameters at STC | IV parameters at STC | 
| -------- | -------- |
| Photocurrent ($I_{ph}$) | Maximum power ($P_{mp}$)|
| Saturation current ($I_{o}$)| Voltage at MPP ($V_{mp}$)| 
| Series resistance ($R_{s}$)| Current at MPP ($I_{mp}$)| 
| Shunt resistance ($R_{sh}$)| Open-circuit voltage ($V_{oc}$)| 
| Diode factor ($n$)| Short-circuit current ($I_{sc}$)| 

## Application
PV-Pro has two major applications:
- **Degradation analaysis**: Calculate the degradation rate of the SDM and IV parameters. See example: [Degradation_analysis.ipynb](examples/Degradation_analysis.ipynb)
- **Irradiance-to-power conversion**: Use the estimated SDM parameters to map the forecasted irradiance to power. See example: [Degradation_analysis.ipynb](examples/Power_prediction.ipynb)

## Package overview

Here's a high level overview of the most important parts of the package.

- preprocess.Preprocessor - class for the pre-processing of data
- main.PvProHandler.run_pipeline - class method to run the parameter estimation
- main.PvProHandler.system_modelling - class method to model the power
- plotting.plot_results_timeseries - function to plot the degradation trend of parameters
- plotting.plot_predicted_ref_power - function to plot the predicted and reference power


# Examples

## Degradation analaysis

The [NIST ground array dataset](https://pvdata.nist.gov/) provides a useful testbed for PV-Pro. A jupyter notebook showing analysis is provided in [Degradation_analysis.ipynb](examples/Degradation_analysis.ipynb). PV-Pro estimates the trend of the SDM and IV parameters over time to interpret what is degrading in the PV system. 

<img src="https://github.com/DuraMAT/pvpro/blob/master/doc_img/NIST_ground_results.png" width="800"/>

From the results, the degradation of power (about -1.29%/yr) is mainly related to the degradation of the current-related parameters ($I_{mp}$, $I_{sc}$, and $I_{ph}$), which notably dropped since 2018. System operators were then advised to pay attention to factors impeding the generation of current, like soiling.

Detailed analysis example (including more figures and post processing) is available [here](examples/Degradation_analysis_detailed.ipynb).

## Irradiance-to-power conversion 

When the forecasted ground weather data is available, PV-Pro can also perform precise irradiance-to-power conversion based on the estimated SDM parameters that reflect the **actual health status** of the PV system.  A jupyter notebook is presented in [Degradation_analysis.ipynb](examples/Power_prediction.ipynb). Here, we focus on a daily power prediction with example results on two days ([NIST dataset](https://pvdata.nist.gov/)) with different weather (clear and cloudy) presented below.

<img src="https://github.com/DuraMAT/pvpro/blob/master/doc_img/power_prediction_results.png" width="700"/>

It is shown that PV-Pro achieves an outstanding power conversion on both days with nMAE <1%.

# Contribution

### Contributors to this content:
Baojie Li, Todd Karin.

### Contributors to PV-Pro project:
Baojie Li (LBL), Todd Karin (PVEL), Bennet E. Meyers (SLAC), Xin Chen (LBL), Dirk C. Jordan (NREL), Clifford W. Hansen (Sandia), Bruce H. King (Sandia), Michael G. Deceglie (NREL), Anubhav Jain (LBL)


