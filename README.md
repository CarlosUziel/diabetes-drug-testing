<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Selection of candidates for a diabetes drug clinical trial using a machine learning regression model on EHR data</h3>

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#premise">Premise</a></li>
        <li><a href="#data">Data</a></li>
        <li><a href="#execution-plan">Execution Plan</a></li>
        <li><a href="#challenges-and-improvements">Challenges and improvements</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#setting-up-a-conda-environment">Setting up a conda environment</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#file-descriptions">File descriptions</a></li>
      </ul>
    </li>
    <li><a href="#additional-notes">Additional Notes</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

## About The Project

In this mini-project, I developed a machine learning regression model to predict the time spent in hospital based on EHR patient data. The ultimate goal is to identify potential candidates for a diabetes drug test, which must be patients expected to stay in the hospital for at least 5 days in order to minimize extra costs in the drug trial.

### Premise

You are a data scientist for an exciting unicorn healthcare startup that has created a groundbreaking diabetes drug that is ready for Phase III clinical trial testing. It is a very unique and sensitive drug that requires administering and screening the drug over at least 5-7 days of time in the hospital with frequent monitoring/testing and patient medication adherence training with a mobile application. You have been provided a patient dataset from a client partner and are tasked with building a predictive model that can identify which type of patients the company should focus their efforts testing this drug on. Target patients are people that are likely to be in the hospital for this duration of time and will not incur significant additional costs for administering this drug to the patient and monitoring.  

In order to achieve your goal you must build a regression model that can predict the estimated hospitalization time for a patient and use this to select/filter patients for your study.

**Expected Hospitalization Time Regression Model:** Utilizing a synthetic dataset (denormalized at the line level augmentation) built off of the UCI Diabetes readmission dataset, students will build a regression model that predicts the expected days of hospitalization time and then convert this to a binary prediction of whether to include or exclude that patient from the clinical trial.

This project will demonstrate the importance of building the right data representation at the encounter level, with appropriate filtering and preprocessing/feature engineering of key medical code sets. This project will also require students to analyze and interpret their model for biases across key demographic groups.

<p align="right">(<a href="#top">back to top</a>)</p>

### Data

Due to healthcare PHI regulations (HIPAA, HITECH), there are limited number of publicly available datasets and some datasets require training and approval. So, for the purpose of this exercise, we are using a [dataset from UC Irvine](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) that has been modified for this course. Please note that it is limited in its representation of some key features such as diagnosis codes which are usually an unordered list in 835s/837s (the HL7 standard interchange formats used for claims and remits).

The dataset reference information can be found [here](https://github.com/udacity/nd320-c1-emr-data-starter/blob/master/project/data_schema_references/). There are two CSVs that provide more details on the fields and some of the mapped values.

<p align="right">(<a href="#top">back to top</a>)</p>

### Execution plan

In order to process the data and build the classification model, the following steps were taken:

1. **Exploratory Data Analysis**: We load, clean, transform and visualize the data as needed.
2. **Model Building**: We setup all the necessary classes and functions to train our pneumonia classifier. This includes choices such as data splits and model hyper-parameters.
3. **Model Uncertainty Analysis**: Obtain mean and standard deviation values of the predictions per patient.
4. **Continous to Binary Convertion**: Convert continuous predictions to binary labels for patient selection.
5. **Model Evaluation**: We use an exhaustive list of performance metrics to validate the predictions during and after training.
6. **Model Biases Evaluation**: Examine possible model biases for race and gender.

### Challenges and improvements

The dataset used for this project was fairly clean and didn't require any elaborate pre-processing. However, some fields, such as medical specialty, which could have been interesting features to learn on, could not be used due to most values being missing.

The model used in this project is a fairly simple one, composed of just a few fully-connected layers and including regularization techniques such as weight normalization, as well as batch normalization and dropout layers. Naturally, the choices for improvement are endless, from using a completely different architecture to tuning each hyper-parameter such as activation functions, optimizers and so on.

It is also worth mentioning that other families of algorithms such as random forests and boosting ensemble algorithms (e.g. XGBoost, LightGBM, etc.) could have also been used. In fact, these are known for achieving state of the art results on tabular data.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Getting Started

To make use of this project, I recommend managing the required dependencies with Anaconda.

### Setting up a conda environment

Install miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Install mamba:

```bash
conda install -n base -c conda-forge mamba
```

Install environment using provided file:

```bash
mamba env create -f environment.yml # alternatively use environment_hist.yml if base system is not debian
mamba activate diabetes_drug_testing
```

### Usage

This project is contained within a main Jupyter notebooks, namely `notebooks/main.ipynb`, containing all the sections of the project.

### File descriptions

The project files are structured as follows:

- `data/schema`: Contains schema and other metadata files.
- `data/vocabulary`: Vocabulary files for the creation of the categorical features.
- `notebooks`: Location of the main project notebook.
- `src/utils.py`: Contains all the utility functions used throughout the project.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Additional Notes

Source files formatted using the following commands:

```bash
isort .
autoflake -r --in-place --remove-unused-variable --remove-all-unused-imports --ignore-init-module-imports .
black .
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

[Carlos Uziel Pérez Malla](https://www.carlosuziel-pm.dev/)

[GitHub](https://github.com/CarlosUziel) - [Google Scholar](https://scholar.google.es/citations?user=tEz_OeIAAAAJ&hl=es&oi=ao) - [LinkedIn](https://at.linkedin.com/in/carlos-uziel-p%C3%A9rez-malla-323aa5124) - [Twitter](https://twitter.com/perez_malla)

## Acknowledgments

This project was done as part of the [AI for Healthcare Nanodegree Program at Udacity](https://www.udacity.com/course/ai-for-healthcare-nanodegree--nd320).

<p align="right">(<a href="#top">back to top</a>)</p>
