class FloatNum:
    """
    Class for representing base10 floating point numbers
    in format of base10 mantissa and base10 exponent.
    The exponent is shifted by a fixed value to prevent usage of negative values
    for negative exponents.

    Attributes
    ----------
    mantissa : int
        unsigned integer value representing the significant digits of the number (max "precision" digits)
    exponent: int
        signed integer value of the exponent
    precision: int
        unsigned integer value representing the number of the significant digits in mantissa
    exp_pad: int
        unsigned integer value for shifting the original exponent (default is 0)
    
    Methods
    -------
    truncate()
        Truncating the value of the mantissa to "precission" number of digits, 
        while updating the exponent
    __add__(num_2) -> FloatNum
        Overrides + operator, adds the value of the FloatNum to the other and returns the resulting FloatNum
    __mul__(num_2) -> FloatNum
        Overrides * operator, multiplies the value of the FloatNum to the other and returns the resulting FloatNum
    __gt__(num_2) -> boolean
        Overrides > operator, returns true if the value of the FloatNum is greater than the other one
    __lt__(num_2) -> boolean
        Overrides > operator, returns true if the value of the FloatNum is less than the other one
    get_noir_input() -> string
        Returns the string representation of the FloatNum object using Noir language struct syntax
    get_prover_input() -> string
        Returns the string representation of the FloatNum object using and array syntax for the Noir language Prover.toml input file
    truncated() -> FloatNum
        Returns truncated value of the FloatNum
    __str__() -> string
        Returns string representation of the FloatNum in scientific notation (<mantissa>e<exponent>)
    """

    def __init__(self, mantissa, exponent, precision, exp_pad=0):
        self.precision = precision
        self.mantissa = mantissa

        # Example:
        # Original exponent = -1
        # exp_pad = 100
        # shifted exponent = 99
        self.exponent = exponent + exp_pad

    def truncate(self):
        tr = self.truncated(self.mantissa, self.exponent)
        self.mantissa = tr.mantissa
        self.exponent = tr.exponent
        return self

    def __add__(self, num_2):
        mant_1 = self.mantissa
        mant_2 = num_2.mantissa

        exp_1 = self.exponent
        exp_2 = num_2.exponent
        exp = exp_1
        diff = abs(exp_1 - exp_2)
        if self.exponent < num_2.exponent:
            mant_2 *= 10 ** diff
            exp = exp_2 - diff
        else:
            mant_1 *= 10 ** diff
            exp = exp_1 - diff

        sum_mant = mant_1 + mant_2

        return self.truncated(sum_mant, exp)

    def __mul__(self, num_2):
        mant_1 = self.mantissa
        mant_2 = num_2.mantissa

        exp_1 = self.exponent
        exp_2 = num_2.exponent

        return self.truncated(mant_1 * mant_2, (exp_1 + exp_2 - 100))

    def __gt__(self, num_2):
        return self.mantissa * (10 ** (self.exponent - 100)) > num_2.mantissa * (10 ** (num_2.exponent - 100))

    def __lt__(self, num_2):
        return self.mantissa * (10 ** (self.exponent - 100)) < num_2.mantissa * (10 ** (num_2.exponent - 100))

    def get_noir_input(self):
        sign = 0
        mant = self.mantissa
        exp = self.exponent
        if mant < 0:
            sign = 1
            mant = -mant

        return 'Float {' f'sign: {sign}, mantissa: {mant}, exponent: {exp}' + ' }'

    def get_prover_input(self):
        sign = 0
        mant = self.mantissa
        exp = self.exponent
        if mant < 0:
            sign = 1
            mant = -mant

        return f'["{sign}", "{mant}", "{exp}"]'

    def truncated(self, mant, exp):
        if len(str(abs(mant))) > self.precision:
            l = len(str(abs(mant)))
            sign_comp = 0
            if mant < 0:
                sign_comp = 1
            prec_diff = abs(l - self.precision)
            mant = int(str(mant)[:self.precision + sign_comp])
            exp += prec_diff

        if mant == 0:
            exp = 100

        return FloatNum(mant, exp, self.precision)

    def __str__(self):
        return f'{self.mantissa}e{self.exponent}'