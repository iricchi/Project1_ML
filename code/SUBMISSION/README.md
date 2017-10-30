# Data analysis and prediction models for the Higgs boson Machine Learning Challenge

## Prerequisites:
No external libraries have been used for this project, in order to run the jupyter notebooks the test.csv and train.csv datasets must be extracted 
from the data.zip file and placed on the same directory as the rest of the code.

## Running the tests:
In this submission folder is present a run.py file that implements the training of our best model and two notebooks that give examples of how we optimized the various methods.

1)'run.py':
	Implements the logistic regression method as explained in the report optimized with various optimization loops such as stepwise and degree optimization, handles feature
	generation and extraction as well as handling the (-999) values. The run.py does not tune the hyperparameters, they have been tuned separately and their values have been
	put into this script in order to not slow it too much. The script produces a .csv that represent our kaggle submission.

2)'optimize_ls_model.ipynb' and 'optimize_lr_model.ipynb':
	Here we present an example of how the optimization process works, the main algorithm used is the stepwise, as explained in the report, since it is very costly it will take 
	hours to find the best features set. We then use cross-validation to evaluate our results and select the optimal degree.
	
## Main functions:
We implemented various important functions to test our ideas, here are explained the most important ones:

1)'stepwise.py': 
	Stepwise function takes in input the model dictionary (containing all the information on the model that should be used) and the R2_method, we decided for the final version to apply,
	as explained in the report, the McFadden Pseudo R2 method. 
    The algorithm consists of a forward elimination: starting with no variables in the model, it tests the addition of each variable using a chosen model fit criterion, 
	adding the variable (if any) whose inclusion gives the most statistically significant improvement of the fit, and repeating this process until none improves the model 
	to a statistically significant extent.    
	
2)'cross_validation.py':
	Here is implemented the standard cross-validation algorithm as seen in the labs, the function works with all the models and take as input a dictonary (args) containing all the infos concerning
	the model to use, we implemented the possibility to output three different losses: rmse, mae, loglikelyhood.
	
3)'implementation_enhanced.py' and 'implementations.py':
	In implementations.py are present the 6 functions requested from the project, instead implementations_enhanced.py is the file we use in our project, we basically added a stopping 
	criteria for the descent methods in order to speed them up.
	
4)'optimize_hyperparams.py':
	Here are present funcitons that implements the opimization loops for the hyperparameters lambda, degree and gamma(although never used the last one). The loop checks the efficacy of each
	value of hyperparameter with the others through cross-validation. The final result is a list of all the results from the cross-validations and the oprimal hyperparameter chosen.