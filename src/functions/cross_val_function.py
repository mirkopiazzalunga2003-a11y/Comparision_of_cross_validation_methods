def cross_validation_function(
    y,                    # y as data frame
    X,                    # X as data frame
    cv_method = None,     # cross validation object
    nested = False,       # definisco se voglio fare nested cv 
    inner_cv = None,      # cross validation object per nested 
    outer_cv = None,   
    param_grid = None,    # come input utilizzare sempre un dizionario di liste 
    seed = 11
    ):

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 5],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2]
        }
    
    # verifico se input di param grid sono correttamente inizializzati
    else: 
        # verifico che param grid sia un dizionario
        if not isinstance(param_grid, dict):
            raise TypeError("param_grid deve essere un dizionario.")

        # verifico che param grid sia un dizionario di liste
        if not all(isinstance(v, list) for v in param_grid.values()):
            raise TypeError("Ogni valore di param_grid deve essere una lista.")

        # verifico che le liste non siano vuote
        if not param_grid or any(not v for v in param_grid.values()):
            raise ValueError("param_grid e le sue liste non possono essere vuote.")

    # se nested è vero vado a far partire ciclo per ottimizzazione iperparametri
    if nested: 
    
        # validi oggetti di cross validation 
        valid_cv = (KFold, StratifiedKFold, ShuffleSplit, LeaveOneOut)

        # verifica corretta inizializzazione inner e outer
        if inner_cv is None or outer_cv is None : 

            # utilizzo ValueError se valore non è incompatibile con requisiti funzione
            raise ValueError("Specificare inner_cv e outer_cv per la nested cross validation")
            
        if not isinstance(inner_cv, valid_cv):
            
            # utilizzo TypeError se oggetto è del tipo sbagliato 
            raise TypeError(f"inner_cv deve essere uno tra {valid_cv}")

        if not isinstance(outer_cv, valid_cv):
            raise TypeError("outer_cv non è di un tipo valido.")

        # considero possibilità di fare nested cv con diversi metodi di cross validation per outer    
        if isinstance(outer_cv,(KFold,ShuffleSplit,LeaveOneOut)): 

            # Definisco l'oggetto per lo split dei dati train-test 
            split_data = outer_cv.split(X)

        # possibile altra alternativa
        elif isinstance(outer_cv,StratifiedKFold): 
            split_data = outer_cv.split(X,y)
        

        # definisco modello
        model = RandomForestClassifier(random_state = seed)

        # definisco grilgia di iperparametri per random forest 
        grid_search = GridSearchCV(
            estimator = model, 
            param_grid = param_grid, 
            cv = inner_cv,                # oggetto che decide come verranno suddivisi i dati di training 
            scoring = "matthews_corrcoef"  # in base a cosa scelgo modello migliore 
                                          # accuracy, roc_auc, mcc (matthews_corrcoef)
        )

        # salvo risultati per calcolare metriche
        y_true = []
        y_pred = []
        y_score = []

        # inizializzo tracker di code carbon
        tracker = EmissionsTracker(
            tracking_mode="process",  # considero processo python e non della macchina 
            measure_power_secs=1,     # considero misurazioni ogni secondo
            log_level= "CRITICAL",    # evito che info/warning
            save_to_file=False
        )
        tracker.start()

        # ciclo per valori definiti dall oggetto method
        for train_index, val_index in split_data: 

            # divido train e test con split oggetto method
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[val_index], y.iloc[val_index]

            grid_search.fit(X_train,y_train)

            best_model = grid_search.best_estimator_

            # per osservare iperparametri migliori
            # grid_search.best_params_

            y_true.append(y_test)
            y_pred.append(best_model.predict(X_test))
            y_score.append(best_model.predict_proba(X_test)[:,1])
        
        # salvo i dati di code carbon
        tracker.stop()
        data = tracker.final_emissions_data

        consumptions = {
            "energy_wh": data.energy_consumed *1000 ,
            "emissions_g_co2eq": data.emissions * 1000 ,
            "duration_s": data.duration
        }

        # concateno valori in un unico array
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)

        # calcolo metriche con la mia funzione 
        metrics = my_metrics(y_true, y_pred, y_score)

        all_results = pd.concat([pd.Series(consumptions),pd.Series(metrics)])
        df_results = pd.DataFrame(all_results).T
        
        return df_results
    
    # considero altre tipologie di cross validation
    else: 

        # verifico se metodo è stato inizializzato
        if cv_method is None:
            raise ValueError(
                "Specificare cv_method quando nested=False."
            )

        # verifico se oggetto method è correttamente inizializzato 
        if isinstance(cv_method,(KFold,ShuffleSplit,LeaveOneOut)) :
            split_data = cv_method.split(X)
        elif  isinstance(cv_method,StratifiedKFold) : 
            split_data = cv_method.split(X,y)
        else : 
            raise TypeError("oggetto cv_method non inizializzato correttamente")

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

            # fitto modello
            model.fit(X_train,y_train)

            y_true.append(y_test)
            y_pred.append(model.predict(X_test))
            y_score.append(model.predict_proba(X_test)[:,1])

        tracker.stop()
        data = tracker.final_emissions_data

        consumptions = {
            "energy_wh": data.energy_consumed *1000 ,
            "emissions_g_co2eq": data.emissions * 1000 ,
            "duration_s": data.duration
        }

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)

        metrics = my_metrics(y_true, y_pred, y_score)

        all_results = pd.concat([pd.Series(consumptions),pd.Series(metrics)])
        df_results = pd.DataFrame(all_results).T
        
        return df_results