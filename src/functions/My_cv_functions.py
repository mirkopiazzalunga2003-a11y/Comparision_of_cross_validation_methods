def my_k_fold_cv(X, k:int, shuffle:bool = True, seed = 11): 
    # X as DataFrame 
    # shuffle considera possibile rimescolamento dell'indice

    n = len(X)
    index = np.arange(n)
    if k < 2  or k > n:
        raise ValueError("valore di k non valido") 

    # considero possibile mescolamento delle osservazioni
    if shuffle: 
        rng = np.random.default_rng(seed)
        rng.shuffle(index)

    # divido array in k sotto array 
    folds = np.array_split(index,k )

    for i in range(k):
        index_val = folds[i]
        # concateno gli array che non sono parte dell'index val
        index_train = np.concatenate([folds[j] for j in range(k) if j !=i ])
        # utilizzo yield per restituire il risultato alla volta permettendo di riprendere la funzione alla prossima iterazioen 
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
    # R rapresenta il numero di folds ripetuti 
    # p rapresenta percentuale di valori del training set 

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

        # restituisce valori presenti nel primo ma non presenti nel secondo  
        index_val = np.setdiff1d(index,index_train)

        yield index_train,index_val


def my_stratified_k_fold_cv(y, k: int, shuffle: bool = True, seed=11):
    # y as DataFrame 
    # shuffle considera possibile rimescolamento dell'indice

    n = len(y)
    index = np.arange(n)

    if k < 2 or k > n:
        raise ValueError("Valore di k non valido")

    classes, counts = np.unique(y, return_counts=True)

    # vincolo su numerosità dei valori della classe meno numerosa
    if np.min(counts) < k:
        raise ValueError("Ogni classe deve avere almeno k osservazioni")

    # creo k fold vuoti
    folds = [[] for _ in range(k)]

    rng = np.random.default_rng(seed)
    for c in classes:

        # estraggo indici di ogni classe
        index_class = np.where(y == c)[0]

        # considero possibile rimescolamento
        if shuffle:
            rng.shuffle(index_class)

        # faccio split degli indici per ottenere k folds
        folds_class = np.array_split(index_class, k)

        # salvo il fold corrispondente
        for j in range(k):
            folds[j].append(folds_class[j])

    # ciclo per indici di training e validation
    for j in range(k):
        index_val = np.concatenate(folds[j])
        index_train = np.setdiff1d(index, index_val)

        yield index_train, index_val


def my_repeated_stratified_hold_out_cv(y,R:int, p :float,seed ): 
    # Y as dataframe
    # R rapresenta il numero di folds ripetuti  
    # p proporzione del training set

    rng = np.random.default_rng(seed)
    for r in range(R):
        # estraggo classi e numerosità di ogni classe
        classes, counts = np.unique(y, return_counts = True)

        index_train = []
        for i,c in enumerate(classes): 
            # definisco indici della classe c 
            index_class = np.where(y == c)[0]
            # definisco proporzione della classe c 
            n_c = p * counts[i]

            # estraggo valori casuali dagli indici della classe c
            index_train.append(rng.choice(index_class, size = int(n_c), replace = False))

        index_train = np.concatenate(index_train)

        index_val = np.setdiff1d(np.arange(len(y)), index_train)

        yield index_train, index_val

