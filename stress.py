import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, cross_val_predict


def main() -> None:
    file = pd.read_csv('dataset/stress.csv')
    print(file.groupby('stress_level').size().reset_index(name='count'))
    file = np.array(file)
    data = file[:, :-1].astype(np.float64)
    classes = file[:, -1]
    metrics = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    nb = GaussianNB()
    scores_nb = cross_validate(nb, data, classes, cv=7, scoring=metrics)
    print(f"\n=== Metrics Bayes ===")
    for s in scores_nb:
        print(f"{s} Mean: {np.average(scores_nb[s])} Standard: {np.std(scores_nb[s])}")
    print("=== Confusion Bayes ===")
    predict_nb = cross_val_predict(nb, data, classes, cv=5)
    matrix_confusion_nb = confusion_matrix(classes, predict_nb)
    print(matrix_confusion_nb)
    disp_nb = ConfusionMatrixDisplay(
        confusion_matrix=matrix_confusion_nb, display_labels=list(set(classes))
    )
    # disp_nb.plot()
    knn = KNeighborsClassifier(n_neighbors=3)
    scores_knn = cross_validate(knn, data, classes, cv=7, scoring=metrics)
    print(f"\n=== Metrics KNN ===")
    for s in scores_knn:
        print(f"{s} Mean: {np.average(scores_knn[s])} Standard: {np.std(scores_knn[s])}")
    print("=== Confusion KNN ===")
    predict_knn = cross_val_predict(knn, data, classes, cv=5)
    matrix_confusion_knn = confusion_matrix(classes, predict_knn)
    print(matrix_confusion_knn)
    disp_knn = ConfusionMatrixDisplay(
        confusion_matrix=matrix_confusion_knn, display_labels=list(set(classes))
    )
    # disp_knn.plot()
    svm = SVC(kernel='poly')
    scores_svm = cross_validate(svm, data, classes, cv=7, scoring=metrics)
    print(f"\n=== Metrics SVM ===")
    for s in scores_svm:
        print(f"{s} Mean: {np.average(scores_svm[s])} Standard: {np.std(scores_svm[s])}")
    print("=== Confusion SVM ===")
    predict_svm = cross_val_predict(svm, data, classes, cv=5)
    matrix_confusion_svm = confusion_matrix(classes, predict_svm)
    print(matrix_confusion_svm)
    disp_svm = ConfusionMatrixDisplay(
        confusion_matrix=matrix_confusion_svm, display_labels=list(set(classes))
    )
    disp_svm.plot()
    plt.show()


if __name__ == '__main__':
    main()
