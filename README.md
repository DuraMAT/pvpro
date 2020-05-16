# PV Production Tools (pvpro)

The typical photovoltaic (PV) analysis tools focus on extracting the rate of change of power at reference conditions. This quantity, *pmp_ref*, is just one of many physical parameters of a PV system. 

Fortunately, in a typical PV dataset, more information is stored than just the DC or AC power. When a dataset also contains the DC voltage, DC current, module temperature and plane-of-array irradiance, we can fit a single-diode model and extract many parameters as a function of time. These parameters include series resistance, shunt resistance, reference photocurrent, and more.

This package, pvpro, automates the analysis of PV production data to extract the rate of change of these parameters. 

**The package is still under active development so don't expect it to work perfectly yet!**

To try it out, you need to clone the most recent development branch of solar-data-tools and statistical-clear-sky.