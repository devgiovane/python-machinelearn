import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier


def old() -> None:
    """"
    Split Test 80/20
    """
    file = pd.read_csv('dataset/iris.csv')
    data = np.array(file)
    characteristics = data[:, :-1].astype(np.float64)
    classes = data[:, -1]
    f_train, f_test, c_train, c_test = train_test_split(characteristics, classes)
    dtc = DecisionTreeClassifier(criterion='entropy')
    dtc.fit(f_train, c_train)
    y_predict = dtc.predict(f_test)
    performance = [0, 0]
    for i in range(len(y_predict)):
        performance[int(y_predict[i] == c_test[i])] += 1
    [error, accuracy] = performance
    print(f"Accuracy: {accuracy} Error: {error}")


def main() -> None:
    """"
    7-Fold Cross Validation
    """
    file = pd.read_csv('dataset/iris.csv')
    data = np.array(file)
    characteristics = data[:, :-1].astype(np.float64)
    classes = data[:, -1]
    dtc = DecisionTreeClassifier(criterion='entropy')
    scores = cross_val_score(dtc, characteristics, classes, cv=10)
    print(f"Mean: {scores.mean()} Standard: {scores.std()}")
    predict = cross_val_predict(dtc, characteristics, classes, cv=10)
    confusion = confusion_matrix(classes, predict)
    print(confusion)


if __name__ == '__main__':
    old()
    main()
