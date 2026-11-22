import os
os.chdir(r"---\\cross_validation")
from src.functions.imports import * 

def cross_validation_function(
    y,                    # y as data frame
    X,                    # X as data frame
    cv_method = None,     # cross-validation object
    nested = False,       # define whether to perform nested CV
    inner_cv = None,      # cross-validation object for nested CV
    outer_cv = None,   
    param_grid = None,    # always use a dictionary of lists as input
    seed = 11
    ):

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 5],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2]
        }
    
    # check whether param_grid inputs are correctly initialized
    else: 
        # check whether param_grid is a dictionary
        if not isinstance(param_grid, dict):
            raise TypeError("param_grid must be a dictorionary.")

        # check whether param_grid is a dictionary of lists
        if not all(isinstance(v, list) for v in param_grid.values()):
            raise TypeError("Every value of param_grid must be a list")

        # check that the lists are not empty
        if not param_grid or any(not v for v in param_grid.values()):
            raise ValueError("param_grid and its list must not be empty.")

    # if nested is True, start the loop for hyperparameter optimization
    if nested: 
    
        # valid cross-validation objects
        valid_cv = (KFold, StratifiedKFold, ShuffleSplit, LeaveOneOut)

        # check that inner and outer are correctly initialized
        if inner_cv is None or outer_cv is None : 

            # use ValueError when a value is incompatible with the function requirements
            raise ValueError("Specify inner_cv and outer_cv for nested cross validation")
            
        if not isinstance(inner_cv, valid_cv):
            
            # use TypeError when the object has the wrong type
            raise TypeError(f"inner_cv must be one of: {valid_cv}")

        if not isinstance(outer_cv, valid_cv):
            raise TypeError("outer_cv is not a valid type.")

        # allow nested CV with different cross-validation methods for the outer loop
        if isinstance(outer_cv,(KFold,ShuffleSplit,LeaveOneOut)): 

            # define the object for splitting the data into training and test sets
            split_data = outer_cv.split(X)

        # another possible alternative
        elif isinstance(outer_cv,StratifiedKFold): 
            split_data = outer_cv.split(X,y)
        

        # define the model
        model = RandomForestClassifier(random_state = seed)

        # define the hyperparameter grid for Random Forest
        grid_search = GridSearchCV(
            estimator = model, 
            param_grid = param_grid, 
            cv = inner_cv,                # object that determines how the training data will be split
            scoring = "matthews_corrcoef"  # criterion used to select the best model
                                          # accuracy, roc_auc, mcc (matthews_corrcoef)
        )

        # store results to calculate metrics
        y_true = []
        y_pred = []
        y_score = []

        # initialize CodeCarbon tracker
        tracker = EmissionsTracker(
            tracking_mode="process",  # track the Python process rather than the entire machine
            measure_power_secs=1,     # collect measurements every second
            log_level= "CRITICAL",    # suppress info/warning messages
            save_to_file=False,
            allow_multiple_runs=True
        )
        tracker.start()

        # loop through the splits defined by the method object
        for train_index, val_index in split_data: 

            # split training and test data using the method object
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[val_index], y.iloc[val_index]

            grid_search.fit(X_train,y_train)

            best_model = grid_search.best_estimator_

            # inspect the best hyperparameters
            # grid_search.best_params_

            y_true.append(y_test)
            y_pred.append(best_model.predict(X_test))
            y_score.append(best_model.predict_proba(X_test)[:,1])
        
        # save CodeCarbon data
        tracker.stop()
        data = tracker.final_emissions_data

        consumptions = {
            "energy_wh": data.energy_consumed *1000 ,
            "emissions_g_co2eq": data.emissions * 1000 ,
            "duration_s": data.duration
        }

        # concatenate values into a single array
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)

        # calculate metrics using my function
        metrics = my_metrics(y_true, y_pred, y_score)

        all_results = pd.concat([pd.Series(consumptions),pd.Series(metrics)])
        df_results = pd.DataFrame(all_results).T
        
        return df_results
    
    # consider other types of cross-validation
    else: 

        # check whether the method has been initialized
        if cv_method is None:
            raise ValueError(
                "Specify cv_method when nested=False."
            )

        # check whether the method object has been correctly initialized
        if isinstance(cv_method,(KFold,ShuffleSplit,LeaveOneOut)) :
            split_data = cv_method.split(X)
        elif  isinstance(cv_method,StratifiedKFold) : 
            split_data = cv_method.split(X,y)
        else : 
            raise TypeError("object cv_method not correctly initialized")

        y_true = []
        y_pred = []
        y_score = []

        tracker = EmissionsTracker(
            tracking_mode="process",  
            measure_power_secs=1,    
            log_level= "CRITICAL",   
            save_to_file=False
        )

        tracker.start()

        for train_index, val_index in split_data:  
            
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[val_index], y.iloc[val_index]

            model = RandomForestClassifier(
                n_estimators = param_grid["n_estimators"][0],            
                criterion = "gini",                                   
                max_depth = param_grid["max_depth"][0],                  
                min_samples_split = param_grid["min_samples_split"][0],  
                min_samples_leaf = param_grid["min_samples_leaf"][0],    
                random_state = seed
            )

            # fit the model
            model.fit(X_train,y_train)

            y_true.append(y_test)
            y_pred.append(model.predict(X_test))
            y_score.append(model.predict_proba(X_test)[:,1])

        tracker.stop()
        data = tracker.final_emissions_data

        consumptions = {
            "energy_wh": data.energy_consumed *1000 ,
            "emissions_g_co2eq": data.emissions *1000 ,
            "duration_s": data.duration
        }

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)

        metrics = my_metrics(y_true, y_pred, y_score)

        all_results = pd.concat([pd.Series(consumptions),pd.Series(metrics)])
        df_results = pd.DataFrame(all_results).T
        
        return df_results
