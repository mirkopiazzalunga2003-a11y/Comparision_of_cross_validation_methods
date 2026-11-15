import numpy as np
def my_metrics(y_true, y_pred, y_score): 

    # ci sarebbero casi limiti da introdurre quando ho denominatori pari a 0

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_score = np.asarray(y_score).ravel()

    # definisco valori confusion metrix 
    TP = np.sum((y_true == 1) & (y_pred ==1))
    TN = np.sum((y_true == 0) & (y_pred ==0))
    FP = np.sum((y_true == 0) & (y_pred ==1))
    FN = np.sum((y_true == 1) & (y_pred ==0))
    n = len(y_true)

    # accuracy 
    # percentuale di valori classificati correttamente rispetto totale 
    acc = (TP + TN) / n

    # sensitivity 
    # percentuale valori positivi predetti correttamente rispetto valori realmente positivi
    sens = TP / (TP + FN)

    # specifity 
    # percentuale valori negativi predetti correttamente rispetto valori realmente negativi
    spec = TN / (TN + FP)

    # precision 
    # percentuale valori positivi predetti correttamente rispetto tutti valori valori predetti positivi 
    prec = TP / (TP + FP)

    # npv 
    # percentuale valori negativi predetti correttamente rispetto tutti valori valori predetti negativi
    npv = TN / (TN + FN)

    # F1 score 
    # sintesi di precision e sensitivity attraverso media armonica per penalizzare valori bassi di almeno uno dei due
    F1 = 2 * (sens * prec) / (sens + prec)

    # mcc
    # misura aggregata di valori confusion metrics, valori vicino a 1 definiscono classificazione migliore 
    MCC = (TP*TN - FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    # Cohen's kappa 
    p_e = ((TP+FN)*(TP+FP) + (TN+FN)*(TN+FP)) / n**2
    if p_e == 1 : 
        k = np.nan
    else : 
        k = (acc - p_e) / (1 - p_e)
    

    # auc è l'area sotto la roc curve 

    # riordino valori score 
    index = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[index]
    y_true_sorted = y_true[index]

    soglie = np.unique(y_score_sorted) # oppure soglie fisse np.linspace(0.01,0.99,10)

    tpr = []
    fpr = []
    precision =[]
    for t in soglie: 
        y_pred_soglia = []

        # costruisco valori di y previsti in base a diverse soglie
        for i in range(len(y_true)): 
            if y_score_sorted[i] >= t: 
                y_pred_soglia.append(1) 
            else: 
                y_pred_soglia.append(0)
        y_pred_soglia = np.array(y_pred_soglia) 

        # calcolo confusion matrix per ogni soglia
        tp = np.sum((y_true_sorted == 1) & (y_pred_soglia ==1)) 
        fp = np.sum((y_true_sorted == 0) & (y_pred_soglia ==1))
        tn = np.sum((y_true_sorted == 0) & (y_pred_soglia ==0))
        fn = np.sum((y_true_sorted == 1) & (y_pred_soglia ==0))
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
        if tp + fp == 0: 
            precision.append(0)
        else:
            precision.append(tp / (tp + fp))

    # aggiungo valori agli estremi
    fpr = np.r_[0, fpr, 1]
    tpr = np.r_[0, tpr, 1]
    precision = np.r_[0, precision, 1]
    recall = tpr.copy()

    # ordino valori di fpr e tpr 
    index = np.argsort(fpr)
    fpr = np.array(fpr)[index]
    tpr = np.array(tpr)[index]

    ROC_AUC = 0
    for i in range(len(fpr)-1): 
        ROC_AUC += (tpr[i] + tpr[i+1])*(fpr[i+1]-fpr[i]) / 2 
    # oppure ROC_AUC = np.trapezoid(tpr, x=fpr)

    # riordino valori per tpr
    index = np.argsort(recall)
    precision = np.array(precision)[index]
    recall = np.array(recall)[index]
    
    PR_AUC = 0
    for i in range(len(recall)-1):  
        PR_AUC += (precision[i] + precision[i+1])*(recall[i+1]-recall[i]) / 2 

    r= 10
    ret = {
        "accuracy":np.round(acc,r), 
        "sensitivity":np.round(sens,r),
        "specifity":np.round(spec,r),
        "precision":np.round(prec,r),
        "npv":np.round(npv,r),
        "F1 score":np.round(F1,r),
        "MCC":np.round(MCC,r),
        "Cohen's kappa":np.round(k,r),
        "ROC AUC":np.round(ROC_AUC,r),
        "PR AUC":np.round(PR_AUC,r) 
       }
       
    return(ret)
