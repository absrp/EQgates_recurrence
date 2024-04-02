# EQgates_recurrence
A Jupyter Notebook to test the effect of earthquake gates on surface rupture length

<!-- ABOUT THE PROJECT -->
## About The Project
Propagating earthquakes must overcome geometrical complexity on fault networks to grow into large, surface rupturing events. Some bends and step-overs of certain geometrical characteristics have the ability to bring ruptures to arrest, capping the final size of the earthquake. This Notebook uses step-overs and bends mapped at length scales of ~100-500 meters from the surface ruptures of 31 strike-slip earthquakes to estimate the rupture passing probability of these features as a function of geometry, and to characterize the relationship between event likelihood and surface rupture length given the distribution of geometrical complexity on a fault. This is a daughter repository to [this project](https://github.com/absrp/passing_probabilities_EQgates). See the link for more information on the raw data and the complete code base to generate the measurements used as input in this Notebook. These data and code were generated as part of Rodriguez Padilla et al. (202X). 

<!-- GETTING STARTED -->
## Data access
The data required to run this Notebook are stored in the csv file "aEQgate_geometry.csv", included in this repository. 

## Prerequisites for running the scripts

All Python packages required to run this Notebook are included in the yaml file included. To create a conda environment to run this Notebook in: 

In your terminal, in the directory where you wish to make the environment and run the code:

```
conda env create -f EQgates_environment.yml
```

All functions called in the Notebook are stored in the function_file.py script, also included in this repository. 

<!-- CONTACT -->
## Contact

Please report suggestions and issues:

Email: alba@caltech.edu

Project Link: [[https://github.com/absrp/passing_probabilities_EQgates](https://github.com/absrp/EQgates_recurrence](https://github.com/absrp/EQgates_recurrence/)/)(https://github.com/absrp/EQgates_recurrence/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
