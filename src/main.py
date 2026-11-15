import os
os.chdir(r"C:\\Users\\mirko\\Desktop\\cross-validation-research")
from src.functions.imports import * 

df = pd.read_csv("data\\Utrecht2019_dataset_plos_one_0222717_s001.csv", sep = "," )

# considero ultima colonna come targhet 
target_name = df.columns[-1]

# shuffle righe e colonne 
# righe 
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# colonne
cols = list(df.columns)
random.shuffle(cols)
df = df[cols]

# Separazione X e y
y = df[target_name]
X = df.drop(columns=[target_name])

seed = 11

# k fold 
k = 5
kf = KFold(
    n_splits=k,        # Numero di fold 
    shuffle=True,      # Mescola casualmente le osservazioni prima di dividerle
    random_state=seed  
)

# repeated hold out 
R = 5 
p = 0.8
repeat_hold_cv = ShuffleSplit(
    n_splits = R,
    train_size = p, 
    random_state = seed
) 

# stratified k fold
strat_kf = StratifiedKFold(
    n_splits=k,
    shuffle=True,
    random_state=seed
)

# leave one out 
loocv = LeaveOneOut()
# loocv = KFold(n_splits = len(X))

# nested k fold 
inner_k = 5 
outer_k = 5
# considero inner_k per identificare migliori iperparametri, si potrebbe anche usare StratifiedKFold
inner_cv = KFold(n_splits = inner_k,shuffle = True, random_state = seed)

# considero outer per osservare performance out of sample
outer_cv = KFold(n_splits = outer_k,shuffle = True, random_state = seed*5)

param_grid = {
    "n_estimators": [100,200],
    "max_depth":[None,5,10],
    "min_samples_split": [2,4,6],
    "min_samples_leaf":[1,2,5]
}

df_k_fold_results = cross_validation_function(y,X,cv_method=kf)

df_repeat_hold_results = cross_validation_function(y,X,cv_method=repeat_hold_cv)

df_loocv_results = cross_validation_function(y,X,cv_method=loocv)

df_strat_kf_results = cross_validation_function(y,X,cv_method=strat_kf)

df_nested_results = cross_validation_function(y,X,nested = True, inner_cv = inner_cv,outer_cv = outer_cv, param_grid = param_grid)

# averaging e aggregazione
df_aggregate = pd.concat([
    df_k_fold_results,
    df_repeat_hold_results,
    df_loocv_results,
    df_strat_kf_results,
    df_nested_results
], ignore_index=True)

df_aggregate.insert(0, "Method", [
    "K-Fold",
    "Repeated Hold-Out",
    "LOOCV",
    "Stratified K-Fold",
    "Nested CV"
])

# salvo df in results
df_aggregate.to_csv("results//summary_df_Utrecht2019.csv",index = False)

