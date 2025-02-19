import argparse
from line_profiler import LineProfiler
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from skproof.mlp.MLPClassifierProver import MLPClassifierProver
print('Loading modules...')


print('Loading dataset...')

# Load test data
iris = load_iris()
X = iris.data
y = iris.target

print('Training MLPClassifier...')

parser = argparse.ArgumentParser(description='Description of your script.')
parser.add_argument('integers', type=int, nargs='+',
                    help='Tuple of integers separated by spaces.')

args = parser.parse_args()
integer_list = list(args.integers)
# print('Hidden layers:', integer_list)

# Train classifier
mlp = MLPClassifier(integer_list, activation='relu', max_iter=5000)
mlp.fit(X, y)

mlpcp = MLPClassifierProver(
    mlp,
    'src/main.nr',
    'Prover.toml',
    './ZKFloat/zkfloat.nr',
    7
)


def proving(M):
    prove_data = M[0, :]
    for idx in range(len(integer_list) + 1):
        # print("prove_data", prove_data)
        out_data = mlpcp.prove(prove_data, idx)
        prove_data = out_data

# Generate proof for the first row
# mlpcp = MLPClassifierProver(
#     mlp,
#     'src/main.nr',
#     'Prover.toml',
#     './ZKFloat/zkfloat.nr',
#     7
# )

# mlpcp.prove(X[:1,:])


lp = LineProfiler()
lp_wrapper = lp(proving)
lp_wrapper(X)
lp.print_stats()
