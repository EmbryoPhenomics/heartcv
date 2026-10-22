# HeartCV Validation and Annotation App

We developed a WebApp for annotating videos to produce ground truth data to validate the performance of HeartCV. The app is built using Dash in Python, and runs in the browser. Below are details on how to install it and use it. 

## Installation

### 1. Clone or download this repository

You can clone this repository using `git` in the terminal using the following command: 

```bash
git clone https://github.com/EmbryoPhenomics/heartcv.git
```

Or simply download and unzip the source code in the latest release in the `releases` section on the GitHub repository: https://github.com/EmbryoPhenomics/heartcv/releases

*Complete all of the following steps in your local copy of this repository, e.g. installation of python requirements.*

### 2. Setting up a Python environment

We recommend using [MiniConda](https://docs.conda.io/projects/miniconda/en/latest/) as your python environment for easy installation of the key dependencies for *HeartCV*. To set up MiniConda please follow these steps:

**Install Python if you don't have it**

For installers for Windows, MacOs and Linux, please head over to the following website: https://www.python.org/downloads/. These installers will walk you through the installation process for your system. 

**Install MiniConda**

To download and install MiniConda, head over to the following documentation: https://docs.conda.io/projects/miniconda/en/latest/. There will be both installers which will walk you through the installation process, but also command line instructions at the bottom of the page if you prefer installing via the terminal.

**Setting up a MiniConda environment**

Once the installation step above has completed, open a terminal window and run the following commands:

Create the Conda environment:
```bash
conda create -n heartcv python=3.12
```
*Note the Python version needs to be between 3.11 and 3.14, see https://www.tensorflow.org/install/pip#software_requirements. We have specified to 3.12 here*

Once completed you can enter the environment like so:
```bash
conda activate heartcv
```

and exit the environment like this: 
```bash
conda deactivate heartcv
```

To install the dependencies of `HeartCV` using the following command in your conda environment:
```bash
conda activate heartcv # Enter the environment
pip3 install -r requirements.txt
```

*Note you will need to be in the `heartcv/validate` folder for the above command to work correctly*

## Usage

To open the app you need to open a terminal in the `heartcv/validate/validation_app` folder, and type in the following command:

```bash
python3 app.py
```

Once you have pressed enter, this command will launch the WebApp and provide you an IP address to enter in your browser. 

After you have opened the WebApp then you will be met with the following interface:

![Alt Text](https://github.com/EmbryoPhenomics/heartcv/blob/main/validate/validation_app/assets/interface_initial.png)


To use the app, the first step you should take is to type in the full file path to the text input at the top left. Once you have, you can click `Record Cardiac Cycle`. This should open up a video interface, separate to the WebApp that will look somewhat like this:

![Alt Text](https://github.com/EmbryoPhenomics/heartcv/blob/main/validate/validation_app/assets/interface.png)

Details on how to operate this interface are presented under the `Frame recorder usage` section in the WebApp. You can use the keys on your keyboard scan through the video and record Diastole or Systole events. The data table and associated cardiac statistics presented at the bottom will only be updated after you have exited the video interface via pressing the `Esc` key. 

To save these cardiac event data, simply type in a file path to where you would like these saved, and press the `Save` button. You will have two files saved, one containes the times of individual beats recorded and the second contains the cardiac statistics. If you wanted to reload data for the same video and pick up where you left off, you can press the `Load` button. Note that in this case, you should set the overwrite check to `No` before clicking `Record Cardiac Cycle` button otherwise the data that you have loaded in the table will be lost.

