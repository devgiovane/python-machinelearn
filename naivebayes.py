import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main() -> None:
    file = pd.read_csv('dataset/iris.csv')
    file = np.array(file)
    data = file[:, :-1].astype(np.float64)
    classes = file[:, -1]
    nb = GaussianNB()
    metrics = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    scores_nb = cross_validate(nb, data, classes, cv=5, scoring=metrics)
    print(f"=== Metrics Bayes ===")
    for s in scores_nb:
        print(f"{s} Mean: {np.average(scores_nb[s])} Standard: {np.std(scores_nb[s])}")
    print("=== Confusion Bayes ===")
    predict_nb = cross_val_predict(nb, data, classes, cv=5)
    matrix_confusion_nb = confusion_matrix(classes, predict_nb)
    print(matrix_confusion_nb)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix_confusion_nb, display_labels=list(set(classes))
    )
    disp.plot()
    plt.show()


if __name__ == '__main__':
    main()
