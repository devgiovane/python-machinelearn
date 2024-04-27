import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main() -> None:
    file = pd.read_csv('dataset/iris.csv')
    file = np.asarray(file)
    data = file[:, :-1].astype(np.float64)
    classes = file[:, -1]
    knn = KNeighborsClassifier(n_neighbors=3)
    metrics = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    scores_knn = cross_validate(knn, data, classes, cv=5, scoring=metrics)
    print(f"=== Metrics KNN ===")
    for s in scores_knn:
        print(f"{s} Mean: {np.average(scores_knn[s])} Standard: {np.std(scores_knn[s])}")
    print("=== Confusion KNN ===")
    predict_knn = cross_val_predict(knn, data, classes, cv=5)
    matrix_confusion_knn = confusion_matrix(classes, predict_knn)
    print(matrix_confusion_knn)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix_confusion_knn, display_labels=list(set(classes))
    )
    disp.plot()
    plt.show()


if __name__ == '__main__':
    main()
