# MicAlgElo

Repository for the paper "From Chess to Academia: Adopting the Elo Rating System for Evaluating Algorithmic Performance" providing a prototype of the proposed tool as well as the required code to run the simulation study described in the paper.

## Description

In the evolving landscape of machine learning (ML), selecting the most effective algorithm for a given use case remains a critical challenge characterized by fuzzy and incomplete information. This paper addresses this challenge by reconceptualizing it as a top-k ranking problem. Traditional approaches to paper rankings are transformed through a methodology based on pairwise comparisons, drawing from established literature on the top-k ranking problem. Our study introduces an innovative adjustment to the widely utilized Elo rating system, enhancing it with the capability to incorporate a weighted combination of multiple confidence metrics. This novel adaptation facilitates the development of a model-driven Decision Support System (DSS) that leverages this adjusted Elo rating algorithm. Comparative results from simulation studies indicate that our proposed DSS outperforms existing methodologies and aligns closely with expert-generated rankings. The findings underscore the utility of our approach in enhancing decision-making in dynamic and complex information environments and contribute a new algorithmic perspective on integrating confidence metrics into Elo ratings. Future work will aim at further real-world validation and enhancement of the DSS to better meet user requirements and integrate more sophisticated features.

## Getting Started

### Dependencies

* Python 3.7

### Installing

* After downloading the repository, install the required packages by running the following command in the terminal:
``` 
pip install requirements -r requirements.txt
```
* For the web app to be working it needs to have access to Google Cloud bucket storage, thus a json file with a user able to access this storage needs to be added to the directory (google-creds.json).
* Having done that, the webapp can be started by running the following command: 
```
python main.py
```
To deploy your own version of the app to Google's App Engine, run the following command:
```
gcloud app deploy
```

## Web App
This Web App is a prototype of the proposed tool. It allows users to upload a dataset of algorithm performance metrics and compare the performance of the algorithms using the proposed Elo rating system. All code for the web app can be found in the main directory of the repository.

A deployed version of the web app can be found [here](https://algelo-algorithmranking.appspot.com/).

## Simulation Study
The simulation study described in the paper can be found in the './experiments' directory and a seperate README.md file is provided to guide the user through the process of running the simulation study.