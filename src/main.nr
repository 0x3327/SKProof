// Struct representing float numbers using sign, mantissa and exponent.
// When Noir language gets the update to support signed integers, the sign field will be removed
struct Float {
    sign: Field,
    mantissa: Field,
    exponent: Field,
}

// Float number precision of mantissa
global precision : Field = 7;

// ReLU activation function used for neural network ML models
fn relu(x : Float) -> Float {
    let mut res = x;
    if x.sign as u64 == 1 {
        res = Float { sign: 0, mantissa: 0, exponent: 100 };
    } 

    res
}

// Truncate Float to "precision" number of digits, 5 in the example
fn truncate(num: Float) -> Float {
    let lookup : [Field; 25] = [
        1,
        10,
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000,
        10000000000,
        100000000000,
        1000000000000,
        10000000000000,
        100000000000000,
        1000000000000000,
        10000000000000000,
        100000000000000000,
        1000000000000000000,
        10000000000000000000,
        100000000000000000000,
        1000000000000000000000,
        10000000000000000000000,
        100000000000000000000000,
        1000000000000000000000000,
    ];

    let maxValue : Field = 10.pow_32(precision);
    let mut decValue : Field = 1;
    let mut logValue : Field = 0;

    for i in 0..25 {
        if num.mantissa as u64 >= lookup[i] as u64 {
            decValue = lookup[i];
            logValue = i;
        }  
    }

    decValue *= 10;
    logValue += 1;

    let mut res : Float = Float { sign: num.sign, mantissa: num.mantissa, exponent: num.exponent };

    if logValue as u64 > precision as u64 {
        let diff = (decValue / maxValue) as u64;
        res = Float { sign: num.sign, mantissa: (num.mantissa as u64 / diff) as Field, exponent: num.exponent + (logValue - precision)};  // 
    }

    if res.mantissa == 0 {
        res = Float { sign: res.sign, mantissa: 0, exponent: 100 };
    }

    res
}

// Multiplication of Float numbers
fn mulFloats(x : Float, y : Float) -> Float {
    let mant = x.mantissa * y.mantissa;
    let exp = x.exponent + y.exponent - 100;
    let mut sign : Field = 0;

    if x.sign != y.sign {
        sign = 1;
    }

    truncate(Float { sign: sign, mantissa: mant, exponent: exp })
}

// Dividing of Float numbers
fn divFloats(x : Float, y: Float) -> Float {

    assert(y.mantissa > 0);

    let mut exp1: Field = x.exponent;
    let mut mant1: u64 = x.mantissa as u64;
    
    let exp2: Field = y.exponent;
    let mant2: Field = y.mantissa;

    // Can't divide lower by higher number with same precision, result will be 0
    // The lower must be multiplied by 10, it means at the same time exponent must be reduced by 1
    if mant1 < mant2 as u64 {
        mant1 *= 10;
        exp1 -= 1;
    }

    let mut new_mant: u64 = 0;
    for i in 0..7 {
        let div = mant1 / mant2 as u64;
        mant1 = (mant1 - mant2 as u64 * div) * 10;
        
        // For precision N, the highest exponent is 10^(N-1)
        let exp = precision - i - 1;
        let pow = 10.pow_32(exp) as u64;
        new_mant += div * pow;
    }

    let new_exp = 100 + exp1 - exp2 - precision + 1;

    let mut new_sign : Field = 0;

    if x.sign as u64 != y.sign as u64 {
        new_sign = 1;
    }

    Float{sign: new_sign, mantissa: new_mant as Field, exponent: new_exp}
}

// Sumation of Float numbers
fn addFloats(x : Float, y : Float) -> Float {
    let mut mant_1 = x.mantissa;
    let mut mant_2 = y.mantissa;

    let mut exp_1 = x.exponent;
    let mut exp_2 = y.exponent;

    let mut diff : Field = 0;
    
    if exp_1 as u64 > exp_2 as u64 { 
        diff = exp_1 - exp_2;
    } else {
        diff = exp_2 - exp_1;
    }

    let mut pow10 : Field = 10.pow_32(diff);

    if x.exponent as u64 < y.exponent as u64 {
      mant_2 *= pow10;
      exp_1 = x.exponent;
    } else {
      mant_1 *= pow10;
      exp_1 = y.exponent;
    }

    let mut sum_mant = mant_1 + mant_2;
    let mut sign = x.sign;

    if x.sign != y.sign {
        if mant_1 as u64 > mant_2 as u64 {
            sum_mant = mant_1 - mant_2;
        } else {
            sum_mant = mant_2 - mant_1;
            sign = y.sign;
        }
    }
    
    truncate(Float { sign: sign, mantissa: sum_mant, exponent: exp_1 })
}

// Subtraction of float numbers
fn subFloats(x : Float, y : Float) -> Float {
    addFloats(x, Float { sign: 1 - y.sign, mantissa: y.mantissa, exponent: y.exponent })
}
fn main(
	x_1 : pub [Field; 3],
	x_2 : pub [Field; 3],
	x_3 : pub [Field; 3],
	x_4 : pub [Field; 3],
	y_1 : pub [Field; 3],
	y_2 : pub [Field; 3],
	y_3 : pub [Field; 3]
) {
	let mut X_0_0 = Float{ sign: x_1[0], mantissa: x_1[1], exponent: x_1[2] };
	let mut X_0_1 = Float{ sign: x_2[0], mantissa: x_2[1], exponent: x_2[2] };
	let mut X_0_2 = Float{ sign: x_3[0], mantissa: x_3[1], exponent: x_3[2] };
	let mut X_0_3 = Float{ sign: x_4[0], mantissa: x_4[1], exponent: x_4[2] };
	let A_1_0_0 = mulFloats(X_0_0, Float {sign: 0, mantissa: 3642836, exponent: 93 });
	let A_1_0_1 = mulFloats(X_0_1, Float {sign: 0, mantissa: 7059380, exponent: 93 });
	let A_1_0_2 = mulFloats(X_0_2, Float {sign: 1, mantissa: 4280614, exponent: 93 });
	let A_1_0_3 = mulFloats(X_0_3, Float {sign: 1, mantissa: 1022961, exponent: 94 });
	let B_0_0_0 = addFloats(A_1_0_0, A_1_0_1);
	let B_0_0_1 = addFloats(B_0_0_0, A_1_0_2);
	let B_0_0_2 = addFloats(B_0_0_1, A_1_0_3);
	let WXb_0_0 = addFloats(B_0_0_2, Float {sign: 0, mantissa: 1745604, exponent: 94 });
	let X_1_0 = relu(WXb_0_0);
	let A_1_1_0 = mulFloats(X_0_0, Float {sign: 1, mantissa: 4393756, exponent: 93 });
	let A_1_1_1 = mulFloats(X_0_1, Float {sign: 1, mantissa: 2092089, exponent: 93 });
	let A_1_1_2 = mulFloats(X_0_2, Float {sign: 0, mantissa: 1577222, exponent: 94 });
	let A_1_1_3 = mulFloats(X_0_3, Float {sign: 0, mantissa: 1838304, exponent: 94 });
	let B_0_1_0 = addFloats(A_1_1_0, A_1_1_1);
	let B_0_1_1 = addFloats(B_0_1_0, A_1_1_2);
	let B_0_1_2 = addFloats(B_0_1_1, A_1_1_3);
	let WXb_0_1 = addFloats(B_0_1_2, Float {sign: 1, mantissa: 2781400, exponent: 93 });
	let X_1_1 = relu(WXb_0_1);
	let A_2_0_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 6706357, exponent: 93 });
	let A_2_0_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 8259019, exponent: 93 });
	let B_1_0_0 = addFloats(A_2_0_0, A_2_0_1);
	let WXb_1_0 = addFloats(B_1_0_0, Float {sign: 0, mantissa: 1101091, exponent: 94 });
	let X_2_0 = relu(WXb_1_0);
	let A_2_1_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 4349370, exponent: 93 });
	let A_2_1_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 5880102, exponent: 93 });
	let B_1_1_0 = addFloats(A_2_1_0, A_2_1_1);
	let WXb_1_1 = addFloats(B_1_1_0, Float {sign: 0, mantissa: 8955358, exponent: 93 });
	let X_2_1 = relu(WXb_1_1);
	let A_2_2_0 = mulFloats(X_1_0, Float {sign: 1, mantissa: 9457559, exponent: 93 });
	let A_2_2_1 = mulFloats(X_1_1, Float {sign: 0, mantissa: 6776603, exponent: 93 });
	let B_1_2_0 = addFloats(A_2_2_0, A_2_2_1);
	let WXb_1_2 = addFloats(B_1_2_0, Float {sign: 1, mantissa: 2013656, exponent: 94 });
	let X_2_2 = relu(WXb_1_2);
	let A_3_0_0 = mulFloats(X_2_0, Float {sign: 0, mantissa: 5701508, exponent: 93 });
	let A_3_0_1 = mulFloats(X_2_1, Float {sign: 0, mantissa: 9337157, exponent: 93 });
	let A_3_0_2 = mulFloats(X_2_2, Float {sign: 1, mantissa: 3986238, exponent: 93 });
	let B_2_0_0 = addFloats(A_3_0_0, A_3_0_1);
	let B_2_0_1 = addFloats(B_2_0_0, A_3_0_2);
	let WXb_2_0 = addFloats(B_2_0_1, Float {sign: 1, mantissa: 1734536, exponent: 94 });
	let O_3_0 = relu(WXb_2_0);
	let A_3_1_0 = mulFloats(X_2_0, Float {sign: 1, mantissa: 5281347, exponent: 93 });
	let A_3_1_1 = mulFloats(X_2_1, Float {sign: 1, mantissa: 5823623, exponent: 93 });
	let A_3_1_2 = mulFloats(X_2_2, Float {sign: 1, mantissa: 1585450, exponent: 94 });
	let B_2_1_0 = addFloats(A_3_1_0, A_3_1_1);
	let B_2_1_1 = addFloats(B_2_1_0, A_3_1_2);
	let WXb_2_1 = addFloats(B_2_1_1, Float {sign: 0, mantissa: 5673855, exponent: 93 });
	let O_3_1 = relu(WXb_2_1);
	let A_3_2_0 = mulFloats(X_2_0, Float {sign: 1, mantissa: 6719779, exponent: 93 });
	let A_3_2_1 = mulFloats(X_2_1, Float {sign: 1, mantissa: 3747686, exponent: 93 });
	let A_3_2_2 = mulFloats(X_2_2, Float {sign: 0, mantissa: 1828329, exponent: 94 });
	let B_2_2_0 = addFloats(A_3_2_0, A_3_2_1);
	let B_2_2_1 = addFloats(B_2_2_0, A_3_2_2);
	let WXb_2_2 = addFloats(B_2_2_1, Float {sign: 1, mantissa: 1661139, exponent: 94 });
	let O_3_2 = relu(WXb_2_2);

	assert(O_3_0.sign == y_1[0]);
	assert(O_3_0.mantissa == y_1[1]);
	assert(O_3_0.exponent == y_1[2]);
	assert(O_3_1.sign == y_2[0]);
	assert(O_3_1.mantissa == y_2[1]);
	assert(O_3_1.exponent == y_2[2]);
	assert(O_3_2.sign == y_3[0]);
	assert(O_3_2.mantissa == y_3[1]);
	assert(O_3_2.exponent == y_3[2]);
}
// A_1_0_0 = 5100000e94 * 3642836e93 = 1857846e94
// A_1_0_1 = 3500000e94 * 7059380e93 = 2470783e94
// A_1_0_2 = 1400000e94 * -4280614e93 = -5992859e93
// A_1_0_3 = 2000000e93 * -1022961e94 = -2045922e93
// B_0_0_0 = 1857846e94 + 2470783e94 = 4328629e94
// B_0_0_1 = 4328629e94 + -5992859e93 = 3729343e94
// B_0_0_2 = 3729343e94 + -2045922e93 = 3524750e94
// WXb_0_0 = 3524750e94 + 1745604e94 = 5270354e94
// A_1_1_0 = 5100000e94 * -4393756e93 = -2240815e94
// A_1_1_1 = 3500000e94 * -2092089e93 = -7322311e93
// A_1_1_2 = 1400000e94 * 1577222e94 = 2208110e94
// A_1_1_3 = 2000000e93 * 1838304e94 = 3676608e93
// B_0_1_0 = -2240815e94 + -7322311e93 = -2973046e94
// B_0_1_1 = -2973046e94 + 2208110e94 = -764936e94
// B_0_1_2 = -764936e94 + 3676608e93 = -3972752e93
// WXb_0_1 = -3972752e93 + -2781400e93 = -6754152e93
// A_2_0_0 = 5270354e94 * 6706357e93 = 3534487e94
// A_2_0_1 = 0e100 * -8259019e93 = 0e100
// B_1_0_0 = 3534487e94 + 0e100 = 3534487e94
// WXb_1_0 = 3534487e94 + 1101091e94 = 4635578e94
// A_2_1_0 = 5270354e94 * 4349370e93 = 2292271e94
// A_2_1_1 = 0e100 * -5880102e93 = 0e100
// B_1_1_0 = 2292271e94 + 0e100 = 2292271e94
// WXb_1_1 = 2292271e94 + 8955358e93 = 3187806e94
// A_2_2_0 = 5270354e94 * -9457559e93 = -4984468e94
// A_2_2_1 = 0e100 * 6776603e93 = 0e100
// B_1_2_0 = -4984468e94 + 0e100 = -4984468e94
// WXb_1_2 = -4984468e94 + -2013656e94 = -6998124e94
// A_3_0_0 = 4635578e94 * 5701508e93 = 2642978e94
// A_3_0_1 = 3187806e94 * 9337157e93 = 2976504e94
// A_3_0_2 = 0e100 * -3986238e93 = 0e100
// B_2_0_0 = 2642978e94 + 2976504e94 = 5619482e94
// B_2_0_1 = 5619482e94 + 0e100 = 5619482e94
// WXb_2_0 = 5619482e94 + -1734536e94 = 3884946e94
// A_3_1_0 = 4635578e94 * -5281347e93 = -2448209e94
// A_3_1_1 = 3187806e94 * -5823623e93 = -1856458e94
// A_3_1_2 = 0e100 * -1585450e94 = 0e100
// B_2_1_0 = -2448209e94 + -1856458e94 = -4304667e94
// B_2_1_1 = -4304667e94 + 0e100 = -4304667e94
// WXb_2_1 = -4304667e94 + 5673855e93 = -3737281e94
// A_3_2_0 = 4635578e94 * -6719779e93 = -3115005e94
// A_3_2_1 = 3187806e94 * -3747686e93 = -1194689e94
// A_3_2_2 = 0e100 * 1828329e94 = 0e100
// B_2_2_0 = -3115005e94 + -1194689e94 = -4309694e94
// B_2_2_1 = -4309694e94 + 0e100 = -4309694e94
// WXb_2_2 = -4309694e94 + -1661139e94 = -5970833e94
// X_0_0 => 5100000e94
// X_0_1 => 3500000e94
// X_0_2 => 1400000e94
// X_0_3 => 2000000e93
// A_1_0_0 => 1857846e94
// A_1_0_1 => 2470783e94
// A_1_0_2 => -5992859e93
// A_1_0_3 => -2045922e93
// B_0_0_0 => 4328629e94
// B_0_0_1 => 3729343e94
// B_0_0_2 => 3524750e94
// WXb_0_0 => 5270354e94
// X_1_0 => 5270354e94
// A_1_1_0 => -2240815e94
// A_1_1_1 => -7322311e93
// A_1_1_2 => 2208110e94
// A_1_1_3 => 3676608e93
// B_0_1_0 => -2973046e94
// B_0_1_1 => -764936e94
// B_0_1_2 => -3972752e93
// WXb_0_1 => -6754152e93
// X_1_1 => 0e100
// A_2_0_0 => 3534487e94
// A_2_0_1 => 0e100
// B_1_0_0 => 3534487e94
// WXb_1_0 => 4635578e94
// X_2_0 => 4635578e94
// A_2_1_0 => 2292271e94
// A_2_1_1 => 0e100
// B_1_1_0 => 2292271e94
// WXb_1_1 => 3187806e94
// X_2_1 => 3187806e94
// A_2_2_0 => -4984468e94
// A_2_2_1 => 0e100
// B_1_2_0 => -4984468e94
// WXb_1_2 => -6998124e94
// X_2_2 => 0e100
// A_3_0_0 => 2642978e94
// A_3_0_1 => 2976504e94
// A_3_0_2 => 0e100
// B_2_0_0 => 5619482e94
// B_2_0_1 => 5619482e94
// WXb_2_0 => 3884946e94
// O_3_0 => 3884946e94
// A_3_1_0 => -2448209e94
// A_3_1_1 => -1856458e94
// A_3_1_2 => 0e100
// B_2_1_0 => -4304667e94
// B_2_1_1 => -4304667e94
// WXb_2_1 => -3737281e94
// O_3_1 => 0e100
// A_3_2_0 => -3115005e94
// A_3_2_1 => -1194689e94
// A_3_2_2 => 0e100
// B_2_2_0 => -4309694e94
// B_2_2_1 => -4309694e94
// WXb_2_2 => -5970833e94
// O_3_2 => 0e100
