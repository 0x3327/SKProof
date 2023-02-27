print('Loading modules...')

from skproof.mlp.MLPClassifierProver import MLPClassifierProver
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

print('Loading dataset...')

# Load test data
iris = load_iris()
X = iris.data
y = iris.target

print('Training MLPClassifier...')

# Train classifier
mlp = MLPClassifier((2,3), activation='relu', max_iter=2000)
mlp.fit(X, y)

# Generate proof for thr first row
mlpcp = MLPClassifierProver(
    mlp,
    'src/main.nr',
    'Prover.toml',
    '../zkfloat/lib.nr',
    7
)

mlpcp.prove(X[:1,:])