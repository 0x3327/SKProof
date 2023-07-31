# SKProof
## About
SKProof [Python](https://python.org/) library enables generation of execution proofs for machine learning models found in [scikit-learn](https://scikit-learn.org/) library. Current version enable generation of proofs for multilayer perceptron neural network classifier ([MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)) but the following updates will include support for more models. The circuits are designed using [Noir](https://noir-lang.org/) language and the PLONK proofs are generated internally using [Nargo](https://noir-lang.org/getting_started/nargo.html) CLI tool, also used to generate Solidity smart contract verifiers for the models. For representing floating point values, we use our [ZKFloat](https://github.com/0x3327/zkfloat) library, written in [Noir](https://noir-lang.org/) language for handling base 10 floating point values.

### Supported models
- MLPClassifier with ReLU activation function

## How it works
The first step is ML model training, where the model parameters are determined. In case of [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), the model parameters are weights between network nodes. The trained model is decomposed into simple addition and multiplication expression, as well as activation function calls. All expressions are converted into calls of [ZKFloat](https://github.com/0x3327/zkfloat) library methods for arithmetic operations over floating point numbers and written, along with the library code, into [Noir](https://noir-lang.org/) program. A user can select the instance for which the prediction proof should be generated and requests proof generation. The SKProof prover invokes [Nargo](https://noir-lang.org/getting_started/nargo.html) CLI commands to compile the circuit and generate the proof for the prediction.

## Planned improvements
- Code optimization
- Support for more models

## Prerequisites
The library uses [Noir](https://noir-lang.org/) code to generate proofs for the [scikit-learn](https://scikit-learn.org/) models, so it is a requirement that you have installed:
- scikit-learn library; install using pip with command `pip install scikit-learn`
- noir library; Installation instructions can be found [here](https://noir-lang.org/getting_started/nargo_installation)

## Installation
Installing skproof package is done using pip with command `pip install skproof`

## Example
Example using Iris dataset and MLPClassifier
```python
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

# Generate proof for the first row
mlpcp = MLPClassifierProver(
    mlp,
    'src/main.nr',
    'Prover.toml',
    '../zkfloat/zkfloat.nr',
    7
)

mlpcp.prove(X[:1,:])
```