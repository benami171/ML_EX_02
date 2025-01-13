from perceptron import perceptron, prepare_iris_data

# Example usage for Setosa vs Versicolor
X, y = prepare_iris_data('iris.txt', 'Iris-versiolor', 'Iris-virginica')
final_weights, num_mistakes = perceptron(X, y)

print(f"Final weight vector: {final_weights}")
print(f"Number of mistakes: {num_mistakes}")