import os
os.chdir(r"C:\\Users\\mirko\\Desktop\\cross_validation")
from src.functions.imports import * 

def my_k_fold_cv(X, k:int, shuffle:bool = True, seed = 11): 
    # X as DataFrame
    # shuffle allows the indices to be randomly shuffled

    n = len(X)
    index = np.arange(n)
    if k < 2 or k > n:
        raise ValueError("valore di k non valido") 

    # allow observations to be shuffled
    if shuffle: 
        rng = np.random.default_rng(seed)
        rng.shuffle(index)

    # split the array into k sub-arrays
    folds = np.array_split(index, k)

    for i in range(k):
        index_val = folds[i]
        # concatenate the arrays that are not part of the validation indices
        index_train = np.concatenate([folds[j] for j in range(k) if j != i])
        # use yield to return one result at a time, allowing the function to resume at the next iteration
        yield index_train, index_val
    

def my_loocv(X): 
    # X as DataFrame

    n = len(X)

    for i in range(n):
        index_val = X.iloc[i]
        
        index_train = pd.concat([X.iloc[[j]] for j in range(n) if j != i])
         
        yield index_train, index_val


def my_repeated_hold_out_cv(X, R: int, p : float, seed = 11): 
    # X as DataFrame
    # R represents the number of repeated splits
    # p represents the proportion of observations in the training set

    if p >= 1 or p <= 0: 
        raise ValueError("Valore di p non valido")
    if not isinstance(R, int) or R <= 0:
        raise ValueError("R deve essere un intero positivo")

    n = len(X)
    n_train = int(p * n)
    if n_train < 1 or n_train >= n:
        raise ValueError("Dimensione dei set non valida")

    index = np.arange(n)

    rng = np.random.default_rng(seed)
    for i in range(R): 
        index_train = rng.choice(index, size = int(p*n), replace = False)

        # return the values present in the first array but not in the second
        index_val = np.setdiff1d(index, index_train)

        yield index_train, index_val


def my_stratified_k_fold_cv(y, k: int, shuffle: bool = True, seed=11):
    # y as DataFrame
    # shuffle allows the indices to be randomly shuffled

    n = len(y)
    index = np.arange(n)

    if k < 2 or k > n:
        raise ValueError("Valore di k non valido")

    classes, counts = np.unique(y, return_counts=True)

    # constraint on the number of observations in the smallest class
    if np.min(counts) < k:
        raise ValueError("Ogni classe deve avere almeno k osservazioni")

    # create k empty folds
    folds = [[] for _ in range(k)]

    rng = np.random.default_rng(seed)
    for c in classes:

        # extract the indices of each class
        index_class = np.where(y == c)[0]

        # optionally shuffle the indices
        if shuffle:
            rng.shuffle(index_class)

        # split the indices into k folds
        folds_class = np.array_split(index_class, k)

        # store the corresponding fold
        for j in range(k):
            folds[j].append(folds_class[j])

    # loop through the training and validation indices
    for j in range(k):
        index_val = np.concatenate(folds[j])
        index_train = np.setdiff1d(index, index_val)

        yield index_train, index_val


def my_repeated_stratified_hold_out_cv(y, R:int, p :float, seed): 
    # y as DataFrame
    # R represents the number of repeated splits
    # p represents the proportion of the training set

    rng = np.random.default_rng(seed)
    for r in range(R):
        # extract the classes and the number of observations in each class
        classes, counts = np.unique(y, return_counts=True)

        index_train = []
        for i, c in enumerate(classes): 
            # define the indices belonging to class c
            index_class = np.where(y == c)[0]
            # define the proportion of class c
            n_c = p * counts[i]

            # randomly sample indices belonging to class c
            index_train.append(rng.choice(index_class, size=int(n_c), replace=False))

        index_train = np.concatenate(index_train)

        index_val = np.setdiff1d(np.arange(len(y)), index_train)

        yield index_train, index_val
