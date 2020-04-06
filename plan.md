- + auto plot maker py
+ making (also a pipeline for dummy variables and autosearch which to keep) - it had effects
- (write function to zip dec tree feature importance)
- add bias or not? how does it changes the model? or leave one dummy out? but that will be the baseline? what if all left in and bias not calculated?
        >>>>> better to leave one dummy out or make an other_dummy (that'll be left out) with all the districts that are uncorroleted (with feature select?!)
creating district averages and only but as dummy for district that have effect on price?!

- 1. **EDA: **
    - corr matrix
    - scatter plot
    - ourlier search
    - geo graph for latitude (color point by price)
- 2.** Modelling**:     
    **cross val check** + a few grid search 
    - Linear regression with polynomals + lasso/elastic
    - quick SVM check (del later? +branch)
    - Random Forrest + corssval (and test oob) gridsearch
    - **auto search for feature selection** (in churn rate proj file)
    - XGBoost Regression test!!!!
    - what other algorithm can be used? 
    - try a few neuron nets to get better results