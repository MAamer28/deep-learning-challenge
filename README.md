# Deep Learning Model for Alphabet Soup
Module 21 Challenge

## Overview of the Analysis

The purpose of the analysis done in this project is to determine if a neural network (NN) model can be constructed to predict the success of applicants for funding from nonprofit foundation Alphabet Soup. 

Using a dataset procured from Alphabet Soup's business team, a NN model would be constructed and trained on data for more than 34,000 organizations that previously received funding. The dataset offers metadata on each organization including features like application type, affiliation in sectors of industry, government classification, use case for funding, organization type, status, income amount, asking amount, and special considerations for a given application. The target variable is a binary success metric on whether the money was used effectively or not as provided in the dataset. 

As per protocols, the dataset was split using a `train_test_split` function to create a training set to fit the models and a testing set to measure the models' strength. Categorical features were encoded using `pd.get_dummies()`. The features were also scaled to account for differences in each feature's raw weight. 

## Results

* Target variable: `IS_SUCCESSFUL`

* Feature variables:
  * APPLICATION_TYPE
  * AFFILIATION
  * CLASSIFICATION
  * USE_CASE
  * ORGANIZATION
  * STATUS
  * INCOME_AMT
  * SPECIAL_CONSIDERATIONS
  * ASK_AMT

* Removed variables
  * EIN
  * NAME

* Initial model:
  * First hidden layer: 80 neurons, relu activation
  * Second hidden layer: 20 neurons, relu activation
  * Output layer: 1 neuron, sigmoid activation
  * 100 epochs
  
<img width="570" alt="Screenshot 2023-09-26 at 5 32 55 PM" src="https://github.com/MAamer28/deep-learning-challenge/assets/130619866/1729b1c7-8e15-4bab-812c-5e0eb29c987f">

* Optimized model:
  * First hidden layer: 8 neurons, relu activation
  * Second hidden layer: 16 neurons, relu activation
  * Third hidden layer: 21 neurons, relu activation
  * Fourth hidden layer: 10 neurons, sigmoid activation
  * Fifth hidden layer: 5 neurons, sigmoid activation
  * Output layer: 1 neuron, sigmoid activation
  * 50 epochs
    
<img width="574" alt="Screenshot 2023-09-26 at 5 32 36 PM" src="https://github.com/MAamer28/deep-learning-challenge/assets/130619866/e9c7e038-09ed-48b5-b7b5-3715e02d758b">

 
## Summary

Both the original and optimized models failed to exceed 73% accuracy in predicting the success of applicants. Several optimizations were attempted modifying the binning cutoffs of various categorical variables, the number of neurons in hidden layers as well as the number of layers, activation functions, and the number of epochs in training the model. Despite the efforts, accuracy remained hovering between 72% and 73%.

Further analysis would be needed to refine the model, including the use of `KerasTuner` to identify the most effective combination of layers, neurons, epochs, and activation functions to maximize accuracy. `KerasTuner` was not used in the current model but must be used in subsequent optimizations.

Alternatively, supervised models can be tested to determine if they would offer better results. Given the type of features and the target variable, a random forest model would be a good starting point to test alternative substitute models. The models can then be evaluated and metrics compared to select the most effective one for future endeavors.
