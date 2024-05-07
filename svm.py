import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main() -> None:
    file = pd.read_csv('dataset/iris.csv')
    file = np.asarray(file)
    data = file[:, :-1].astype(np.float64)
    classes = file[:, -1]
    svm = SVC(kernel='poly')
    metrics = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    scores_svm = cross_validate(svm, data, classes, cv=5, scoring=metrics)
    print(f"=== Metrics SVM ===")
    for s in scores_svm:
        print(f"{s} Mean: {np.average(scores_svm[s])} Standard: {np.std(scores_svm[s])}")
    print("=== Confusion SVM ===")
    predict_svm = cross_val_predict(svm, data, classes, cv=5)
    matrix_confusion_svm = confusion_matrix(classes, predict_svm)
    print(matrix_confusion_svm)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix_confusion_svm, display_labels=list(set(classes))
    )
    disp.plot()
    plt.show()


if __name__ == '__main__':
    main()
