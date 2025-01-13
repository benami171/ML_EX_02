from perceptron import perceptron, prepare_iris_data

def main():
    # Call perceptron for Setosa vs Versicolor
    X1, y1 = prepare_iris_data('iris.txt', 'Iris-setosa', 'Iris-versicolor')
    final_weights1, num_mistakes1 = perceptron(X1, y1)
    print(f"Setosa vs Versicolor: Final weight vector: {final_weights1}")
    print(f"Number of mistakes: {num_mistakes1}")

    # Call perceptron for Setosa vs Virginica
    X2, y2 = prepare_iris_data('iris.txt', 'Iris-setosa', 'Iris-virginica')
    final_weights2, num_mistakes2 = perceptron(X2, y2)
    print(f"Setosa vs Virginica: Final weight vector: {final_weights2}")
    print(f"Number of mistakes: {num_mistakes2}")

    # Call perceptron for Versicolor vs Virginica
    X3, y3 = prepare_iris_data('iris.txt', 'Iris-versicolor', 'Iris-virginica')
    final_weights3, num_mistakes3 = perceptron(X3, y3)
    print(f"Versicolor vs Virginica: Final weight vector: {final_weights3}")
    print(f"Number of mistakes: {num_mistakes3}")



if __name__ == "__main__":
    main()
