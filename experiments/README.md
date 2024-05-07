# Simulation Study - Algorithm Performance Ranking

This subdirectory contains all the code necessary to run the simulation study described in the paper "From Chess to Academia: Adopting the Elo Rating System for Evaluating Algorithmic Performance". The simulation study is designed to compare the performance of the proposed Elo rating system with other ranking methods.

## Getting Started

### Dependencies

* Python 3.7

### Installing

* After downloading the repository, install the required packages by running the following command in the terminal:
``` 
pip install requirements -r requirements.txt
```
* Next, the the appropriate directory structure needs to be created. This can be done by running the following command:
```
mkdir data
mkdir data/eccd
mkdir data/wdbc
mkdir output
mkdir output/full
mkdir output/papers
```
* After the directories are created, the sample datasets need to be downloaded and put into the corresponding data folders. They can be found under the following links:
  * [ECCD](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  * [WDBC](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
* After that some pre-processing of the data needs to be done. For this please run the following command:
```
python dataPreperation.py
```
* With the data ready, the gold standard needs to be created. This is done by running the following command (depending on your computational resources, this might take a whike):
```
python fullEvaluation.py
```
Next, the data of the simulated papers is created by running the following command:
```
python simulatePapers.py
```
Lastly, we can run the simulation study and compare the performance of the different ranking algorithms visually as well as statistically. This is done by running the following command:
```
python main.py
```