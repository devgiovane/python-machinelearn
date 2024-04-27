import math
import pandas as pd


def entropy(classes: dict) -> float:
    """"
    Entropy H(S) = - ∑ p * log2(p)
    """
    t = 0
    for c in classes.values():
        t += -(c * math.log2(c))
    return t


def information_gain(s, total, classes) -> float:
    """"
    Gain Information G(S, A) = H(S) -  ∑ Sv/S * H(Sv)
    """
    t = s
    for c in classes:
        t += - (sum(c) / total * entropy({'y': c[0] / sum(c), 'n': c[1] / sum(c)}))
    return t


def sum_attr_with_classes(attr, classes):
    result = {}
    attr_df = pd.DataFrame({'attr': attr})
    attr_df.fillna('None', inplace=True)
    attr_count = len(attr_df.groupby('attr'))
    classes_df = pd.DataFrame({'class': classes})
    classes_count = len(attr_df.groupby('class'))
    for _ in range(attr_count):
        value = attr_df.groupby('attr').size().reset_index(name='count').loc[_].values[0]
        p = sum(1 for alt, will_wait in zip(attr, classes) if alt == value and will_wait == 'Yes')
        n = sum(1 for alt, will_wait in zip(attr, classes) if alt == value and will_wait == 'No')
        result.update({value: [p, n]})
    return result


def main() -> None:
    file = pd.read_csv('dataset/restaurant.csv')
    [rows_total, _] = file.shape
    classes = file.groupby('will_wait').size().reset_index(name='count')
    rows_no = classes.loc[0].values[1]
    rows_yes = classes.loc[1].values[1]
    print(rows_no, rows_yes)
    s = entropy({'yes': rows_yes / rows_total, 'no': rows_no / rows_total})
    print(f"Entropy(S): {s}")
    alt = sum_attr_with_classes(file['alt'], file['will_wait'])
    print(f"alt => {alt}")
    gain_alt = information_gain(s, rows_total, [alt['Yes'], alt['No']])
    print(f"Gain(S, alt) = {gain_alt}")


if __name__ == '__main__':
    main()
