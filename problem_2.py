'''
Using Python 3.10.12 on wsl
'''

from perceptron import perceptron, prepare_iris_data
from true_margin import calculate_true_margin

def main():
    # Call perceptron for Setosa vs Versicolor
    X1, y1 = prepare_iris_data('iris.txt', 'Iris-setosa', 'Iris-versicolor')
    final_weights1, num_mistakes1 = perceptron(X1, y1)
    true_margin1 = calculate_true_margin(X1, y1)
    print(f"Setosa vs Versicolor:")
    print(f"Final weight vector: {final_weights1}")
    print(f"Number of mistakes: {num_mistakes1}")
    print(f"True margin: {true_margin1}\n")

    # Call perceptron for Setosa vs Virginica
    X2, y2 = prepare_iris_data('iris.txt', 'Iris-setosa', 'Iris-virginica')
    final_weights2, num_mistakes2 = perceptron(X2, y2)
    true_margin2 = calculate_true_margin(X2, y2)
    print(f"Setosa vs Virginica:")
    print(f"Final weight vector: {final_weights2}")
    print(f"Number of mistakes: {num_mistakes2}")
    print(f"True margin: {true_margin2}\n")


if __name__ == "__main__":
    main()
