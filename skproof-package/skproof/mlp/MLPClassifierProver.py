import numpy as np
from skproof.float_num import FloatNum
import math
import os


class MLPClassifierProver:
    """
    Class for generating noir prover for pre-trained MLPClassifier circuit.

    Attributes
    ----------
    clf : object
        Trained Sci-kit learn MLPClassifier model

    Methods
    -------
    quant(q) -> FloatNum
        Generates FloatNum object with given precision from decimal numbers
    float_from_string(str) -> FloatNum
        Creates FloatNum object with given precision from the number string representation in scientific notation
    generate_statements(expressions, inputs, outputs)
        Generates and outputs Noir language addition, multiplication and relu function call statements
        based on generated expressions and input node labels. Generates "constrain" statements based on the outputs
    generate_expressions() -> (expressions, inputs, outputs)
        Generates expressions from the trained MLPClassifier model with the given precision
    generate_ar_statement(arg_1, op, arg_2) -> string
        Generates FloatNum addition and multiplication statements using Noir language syntax, based on the given argumens and operation
    generate_ac_statement(value, act_func) -> string
        Generates activation function call using Noir language syntax, based on the given argument
    simulate_ann(expressions, inputs, outputs, data)->(y_pred, input_values, output_values)
        Simulates neural network execution based on the given model and input data and outputs the results
    import_lib()
        Outputs Noir language struct Float and methods for computations using Float numbers 
    generate_circuit(X)
        Outputs Noir language circuit for the given input data
    prove(X)
        Generates Noir language circuit and generates proof for the given input data
    """

    def __init__(
        self,
        mlp_classifier_model,
        circuit_output_path,
        prover_output_path,
        float_num_lib_path,
        precision=7,
        exp_pad=100,
        verbose=True
    ):
        self.clf = mlp_classifier_model

        self.circuit_output_path = circuit_output_path
        self.circuit_output = open(circuit_output_path, 'w')

        self.prover_output_path = open(prover_output_path, 'w')
        self.float_num_lib = open(float_num_lib_path, 'r')
        self.precision = precision
        self.exp_pad = exp_pad
        self.verbose = verbose

    def quant(self, q):
        return FloatNum(round(q * 10**((self.precision * 2))), -(2 * self.precision), self.precision, self.exp_pad).truncate()

    def float_from_string(self, str_num):
        [mant, exp] = str_num.split('e')
        return FloatNum(int(mant), int(exp), self.precision)

    def generate_ar_statement(self, res, arg_1, op, arg_2):
        if op == '+':
            return f'let {res} = addFloats({arg_1}, {arg_2});'
        elif op == '*':
            return f'let {res} = mulFloats({arg_1}, {arg_2});'

    def generate_ac_statement(self, res, value, act_func):
        return f'let {res} = {act_func}({value});'

    def generate_statements(self, expressions, inputs, outputs):
        statements = []

        for e in expressions:
            operation = e[0]
            variable = e[1]
            arg_1 = e[2]
            arg_2 = None

            if len(e) > 3:
                arg_2 = e[3]

            if isinstance(arg_1, FloatNum):
                arg_1 = arg_1.get_noir_input()

            if isinstance(arg_2, FloatNum):
                arg_2 = arg_2.get_noir_input()

            if operation == 'SUM':
                sum_statement = self.generate_ar_statement(
                    variable, arg_1, '+', arg_2)
                statements.append(sum_statement)
            elif operation == 'MUL':
                mul_statement = self.generate_ar_statement(
                    variable, arg_1, '*', arg_2)
                statements.append(mul_statement)
            elif operation == 'RELU':
                relu_statement = self.generate_ac_statement(
                    variable, arg_1, 'relu')
                statements.append(relu_statement)
            else:
                raise f'Invalid operation: {operation}'

        declarations = ''

        x_num = 1
        for input in inputs:
            declarations += f'\tlet mut {input} = Float' + '{ ' + \
                f'sign: x_{x_num}[0], mantissa: x_{x_num}[1], exponent: x_{x_num}[2]' + ' };\n'
            x_num += 1

        expression_statements = ''
        for st in statements:
            expression_statements += f'\t{st}\n'

        constrains = ''
        y_num = 1

        main_args_arr = []
        main_fn = 'fn main(\n'

        x_num = 1
        for input in inputs:
            main_args_arr.append(f'\tx_{x_num} : pub [Field; 3]')
            x_num += 1


        for out in outputs:
            main_args_arr.append(f'\ty_{y_num} : pub [Field; 3]')
            constrains += f'\tconstrain {out}.sign == y_{y_num}[0];\n'
            constrains += f'\tconstrain {out}.mantissa == y_{y_num}[1];\n'
            constrains += f'\tconstrain {out}.exponent == y_{y_num}[2];\n'
            y_num += 1

        main_fn += ',\n'.join(main_args_arr)
        main_fn += '\n) {\n'

        self.circuit_output.write(main_fn)
        self.circuit_output.write(declarations)
        self.circuit_output.write(expression_statements)
        self.circuit_output.write('\n')
        self.circuit_output.write(constrains)
        self.circuit_output.write('}\n')

    def generate_expressions(self):
        num_layers = len(self.clf.coefs_)
        expressions = []

        for l in range(num_layers):
            layer = self.clf.coefs_[l]
            ints = self.clf.intercepts_[l]

            for j in range(layer.shape[1]):
                coefs = layer[:, j]
                new_expressions = []

                # Extract weight multiplications
                for i in range(coefs.shape[0]):
                    new_expressions.append(
                        ['MUL', f'A_{l+1}_{j}_{i}', f'X_{l}_{i}', self.quant(coefs[i])])

                # Extract sumations
                partials = []
                for poly_i in range(1, len(new_expressions)):
                    if poly_i == 1:
                        sum_arg = ['SUM', f'B_{l}_{j}_{poly_i-1}',
                                   new_expressions[0][1], new_expressions[1][1]]
                    else:
                        sum_arg = ['SUM', f'B_{l}_{j}_{poly_i-1}',
                                   f'B_{l}_{j}_{poly_i-2}', new_expressions[poly_i][1]]
                    partials.append(sum_arg)

                partials.append(
                    ['SUM', f'WXb_{l}_{j}', f'B_{l}_{j}_{len(new_expressions)-2}', self.quant(ints[j])])

                letter = 'X'
                if l == num_layers - 1:
                    letter = 'O'

                # Add ReLU activation function statement
                activations = [
                    ['RELU', f'{letter}_{l+1}_{j}', f'WXb_{l}_{j}']
                ]

                expressions += new_expressions
                expressions += partials
                expressions += activations

        inputs = [f'X_0_{i}' for i in range(self.clf.coefs_[0].shape[0])]
        outputs = [f'O_{len(self.clf.coefs_)}_{i}' for i in range(
            self.clf.coefs_[-1].shape[1])]

        return (expressions, inputs, outputs)

    def simulate_ann(self, expressions, inputs, outputs, data):
        y_pred = []
        for row in data:
            node_values = {}

            # Init inputs
            for i in range(len(inputs)):
                input_node = inputs[i]
                node_values[input_node] = self.quant(row[i])

            for poly in expressions:
                second_val = None

                if len(poly) == 4:
                    if isinstance(poly[3], FloatNum):
                        second_val = poly[3]
                    else:
                        second_val = node_values[poly[3]]

                if poly[0] == 'SUM':
                    node_values[poly[1]] = node_values[poly[2]] + second_val
                    if self.verbose:
                        self.circuit_output.write(
                            f"// {poly[1]} = {node_values[poly[2]]} + {second_val} = {node_values[poly[2]] + second_val}\n")
                elif poly[0] == 'MUL':
                    # Verbose logs in Noir language comments
                    if self.verbose:
                        self.circuit_output.write(
                            f"// {poly[1]} = {node_values[poly[2]]} * {second_val} = {node_values[poly[2]] * second_val}\n")
                    node_values[poly[1]] = node_values[poly[2]] * second_val
                elif poly[0] == 'RELU':
                    if node_values[poly[2]].mantissa < 0:
                        node_values[poly[1]] = FloatNum(
                            0, 0, self.precision, self.exp_pad)
                    else:
                        node_values[poly[1]] = node_values[poly[2]]

            for key, value in node_values.items():
                if self.verbose:
                    self.circuit_output.write(f"// {key} => {value}\n")

            label = np.argmax([node_values[i] for i in outputs])
            y_pred.append(label)

        return y_pred, [node_values[p] for p in inputs], [node_values[o] for o in outputs]

    def import_lib(self):
        lib_data = self.float_num_lib.read()
        lib_data = lib_data.replace('maxValue : Field = 100000;', f'maxValue : Field = {10 ** self.precision};')
        lib_data = lib_data.replace('maxLogValue : Field = 5;', f'maxLogValue : Field = {self.precision};')
        self.circuit_output.write(lib_data)
        self.circuit_output.write('\n')

    def generate_circuit(self, X):
        if self.verbose:
            print('Generating circuit...')

        # Write library into circuit file
        self.import_lib()

        # Generate expressions from MLPClassifier model
        expressions, inputs, outputs = self.generate_expressions()

        # Generate Noir language statements
        self.generate_statements(expressions, inputs, outputs)

        # Generate prover file
        prover_input_file = open('Prover.toml', 'w')
        for i in range(X.shape[1]):
            value = X[0, i].ravel()[0]
            float_value = self.quant(value).get_prover_input()
            prover_input_file.write(f'x_{i+1} = {float_value}\n')

        _, _, output_values = self.simulate_ann(
            expressions,
            inputs,
            outputs,
            X
        )

        for i in range(len(output_values)):
            o = output_values[i].get_prover_input()
            prover_input_file.write(f'y_{i+1} = {o}\n')

        self.circuit_output.close()
        prover_input_file.close()

        if self.verbose:
            print(f'Circuit generated successfuly in {self.circuit_output_path}')

    def prove(self, X):
        self.generate_circuit(X)
        path = self.circuit_output_path.split('/')[:-2]
        
        if self.verbose:
            print('Generating proof, this may take a while...')
        if len(path) != 0:
            os.system(f'cd {"/".join(path)}')
        os.system("nargo prove mlp-proof")
        print(f'Done!')