import os
os.chdir(r"C:\\Users\\mirko\\Desktop\\cross_validation")
from src.functions.imports import * 

def my_metrics(y_true, y_pred, y_score): 

    # edge cases should be considered when denominators are equal to 0

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_score = np.asarray(y_score).ravel()

    # define confusion matrix values
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    n = len(y_true)

    # accuracy
    # percentage of correctly classified observations out of the total number of observations
    acc = (TP + TN) / n

    # sensitivity
    # percentage of correctly predicted positive values out of all actual positive values
    sens = TP / (TP + FN)

    # specificity
    # percentage of correctly predicted negative values out of all actual negative values
    spec = TN / (TN + FP)

    # precision
    # percentage of correctly predicted positive values out of all predicted positive values
    prec = TP / (TP + FP)

    # npv
    # percentage of correctly predicted negative values out of all predicted negative values
    npv = TN / (TN + FN)

    # F1 score
    # combines precision and sensitivity through the harmonic mean to penalize low values of at least one of the two
    F1 = 2 * (sens * prec) / (sens + prec)

    # mcc
    # aggregate measure based on confusion matrix values; values close to 1 indicate better classification performance
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    # Cohen's kappa
    p_e = ((TP + FN) * (TP + FP) + (TN + FN) * (TN + FP)) / n**2
    if p_e == 1:
        k = np.nan
    else:
        k = (acc - p_e) / (1 - p_e)

    # AUC is the area under the ROC curve

    # sort score values
    index = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[index]
    y_true_sorted = y_true[index]

    soglie = np.unique(y_score_sorted)  # alternatively, use fixed thresholds np.linspace(0.01, 0.99, 10)

    tpr = []
    fpr = []
    precision = []

    for t in soglie:
        y_pred_soglia = []

        # construct predicted y values based on different thresholds
        for i in range(len(y_true)):
            if y_score_sorted[i] >= t:
                y_pred_soglia.append(1)
            else:
                y_pred_soglia.append(0)

        y_pred_soglia = np.array(y_pred_soglia)

        # calculate the confusion matrix for each threshold
        tp = np.sum((y_true_sorted == 1) & (y_pred_soglia == 1))
        fp = np.sum((y_true_sorted == 0) & (y_pred_soglia == 1))
        tn = np.sum((y_true_sorted == 0) & (y_pred_soglia == 0))
        fn = np.sum((y_true_sorted == 1) & (y_pred_soglia == 0))

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

        if tp + fp == 0:
            precision.append(0)
        else:
            precision.append(tp / (tp + fp))

    # add endpoint values
    fpr = np.r_[0, fpr, 1]
    tpr = np.r_[0, tpr, 1]
    precision = np.r_[0, precision, 1]
    recall = tpr.copy()

    # sort fpr and tpr values
    index = np.argsort(fpr)
    fpr = np.array(fpr)[index]
    tpr = np.array(tpr)[index]

    ROC_AUC = 0
    for i in range(len(fpr) - 1):
        ROC_AUC += (tpr[i] + tpr[i + 1]) * (fpr[i + 1] - fpr[i]) / 2
    # alternatively, ROC_AUC = np.trapezoid(tpr, x=fpr)

    # sort values by tpr
    index = np.argsort(recall)
    precision = np.array(precision)[index]
    recall = np.array(recall)[index]

    PR_AUC = 0
    for i in range(len(recall) - 1):
        PR_AUC += (precision[i] + precision[i + 1]) * (recall[i + 1] - recall[i]) / 2

    r = 10
    ret = {
        "accuracy": np.round(acc, r),
        "sensitivity": np.round(sens, r),
        "specifity": np.round(spec, r),
        "precision": np.round(prec, r),
        "npv": np.round(npv, r),
        "F1 score": np.round(F1, r),
        "MCC": np.round(MCC, r),
        "Cohen's kappa": np.round(k, r),
        "ROC AUC": np.round(ROC_AUC, r),
        "PR AUC": np.round(PR_AUC, r)
    }

    return ret
