# Artificial-Neural-Network
- Python code for implementing ANN on given dataset.
- Spyder IDE used.


## Installation Notes
**Keras** can only run on **Python version 3.6** so we need to create a new virtual environment using conda to do that.


Following steps to be run using **Anaconda Prompt.**

**Step 1: **

Update conda

	conda update conda
	conda update --all


**Step 2: **

If you have **NVIDIA DEDICATED GRAPHICS** and want to utilize that for Keras then follow **Steps 4 to 6** in the given link:

https://github.com/antoniosehk/keras-tensorflow-windows-installation


<br>If you want to utilize your CPU instead, then go to the next step.


**Step 3:**

If you want to use your **NVIDIA GPU **then run in Anaconda Prompt:

`conda create -n new_env python=3.6 numpy scipy pandas matplotlib statsmodels scikit-learn spyder keras-gpu`

where `new_env` is the name of the new environment that you create.


If you want to use your **CPU** then run in Anaconda Prompt:

`conda create -n new_env python=3.6 numpy scipy pandas matplotlib statsmodels scikit-learn spyder keras`

where `new_env` is the name of the new environment that you create.


**Step 4:**

Activate your new environment by running the following command in Anaconda Prompt:

`conda activate new_env`


**Step 5:**

Open Anaconda Navigator and in the *Home* tab, you should be able to see a drop down box that gives you an option to switch between *base (root)*  and this new environment.

*Switch* to the new environment if not already done, and run Spyder from there.

Everything should be working now.



**NOTES:**

More commands that may be required to handle the virtual environment.


Command to **deactivate** the virtual environment:

`conda deactivate`

Command to **delete** the virtual environment:

`conda env remove -n new_env` 
