import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict, cross_validate


def main() -> None:
    """"
    Confusion Matrix
    """
    file = pd.read_csv('dataset/cancer.csv')
    data = np.array(file)
    characteristics = data[:, 2:].astype(np.float64)
    classes = data[:, 1]
    dtc = DecisionTreeClassifier(criterion='entropy')
    metrics = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    scores = cross_validate(dtc, characteristics, classes, cv=10, scoring=metrics)
    print(f"=== Metrics Tree ===")
    for s in scores:
        print(f"{s} Mean: {np.average(scores[s])} Standard: {np.std(scores[s])}")
    predict = cross_val_predict(dtc, characteristics, classes, cv=10)
    confusion = confusion_matrix(classes, predict)
    print("=== Confusion Tree ===")
    print(f"{confusion}")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=list(set(classes)))
    disp.plot()
    plt.show()


if __name__ == '__main__':
    main()
