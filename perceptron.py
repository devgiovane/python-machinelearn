class Perceptron(object):

    def __init__(self, entries, learning=0.01, epochs=100):
        self.__leaning = learning
        self.__epochs = epochs
        self.__weights = [0] * (entries + 1)

    @staticmethod
    def activate(result):
        print(f"neuron: {result}")
        return 1 if result >= 0 else 0

    def neuron(self, entries):
        total = self.__weights[0]
        for i in range(len(entries)):
            total += entries[i] * self.__weights[i + 1]
        return total

    def train(self, x: list, y: list):
        for _ in range(self.__epochs):
            print(f"=== Epoch: {_ + 1} ===")
            print(f"weights: {self.__weights}")
            for entries, expected in zip(x, y):
                predict = self.activate(self.neuron(entries))
                print(f"predict: {predict}")
                print(f"expected: {expected}")
                error = expected - predict
                self.__weights[0] += self.__leaning * error
                for i in range(len(entries)):
                    self.__weights[i + 1] += self.__leaning * error * entries[i]

    def predict(self, entries):
        return self.activate(self.neuron(entries))


def main() -> None:
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]
    perceptron = Perceptron(entries=2, learning=0.5, epochs=10)
    perceptron.train(x, y)
    # print("Predict:")
    # for entries, expected in zip(x, y):
    #     predict = perceptron.predict(entries)
    #     print(f"Entry: {entries}, Expected: {expected}, Predicted: {predict}")


if __name__ == '__main__':
    main()
