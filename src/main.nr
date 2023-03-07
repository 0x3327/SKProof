// Struct representing float numbers using sign, mantissa and exponent.
// When Noir language gets the update to support signed integers, the sign field will be removed
struct Float {
    sign: Field,
    mantissa: Field,
    exponent: Field,
}

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

    let maxValue : Field = 10000000;
    let maxLogValue : Field = 7;
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

    if logValue as u64 > maxLogValue as u64 {
        let diff = (decValue / maxValue) as u64;
        res = Float { sign: num.sign, mantissa: (num.mantissa as u64 / diff) as Field, exponent: num.exponent + (logValue - maxLogValue)};  // 
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

    let mut pow10 : Field = 1;

    for i in 0..25 {
        if i == diff {
            pow10 = lookup[i];
        }
    }

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
	let A_1_0_0 = mulFloats(X_0_0, Float {sign: 0, mantissa: 1330295, exponent: 94 });
	let A_1_0_1 = mulFloats(X_0_1, Float {sign: 0, mantissa: 5248411, exponent: 93 });
	let A_1_0_2 = mulFloats(X_0_2, Float {sign: 1, mantissa: 1029153, exponent: 94 });
	let A_1_0_3 = mulFloats(X_0_3, Float {sign: 1, mantissa: 1465364, exponent: 94 });
	let B_0_0_0 = addFloats(A_1_0_0, A_1_0_1);
	let B_0_0_1 = addFloats(B_0_0_0, A_1_0_2);
	let B_0_0_2 = addFloats(B_0_0_1, A_1_0_3);
	let WXb_0_0 = addFloats(B_0_0_2, Float {sign: 0, mantissa: 4351563, exponent: 93 });
	let X_1_0 = relu(WXb_0_0);
	let A_1_1_0 = mulFloats(X_0_0, Float {sign: 0, mantissa: 6988025, exponent: 93 });
	let A_1_1_1 = mulFloats(X_0_1, Float {sign: 1, mantissa: 6373806, exponent: 93 });
	let A_1_1_2 = mulFloats(X_0_2, Float {sign: 0, mantissa: 9324125, exponent: 93 });
	let A_1_1_3 = mulFloats(X_0_3, Float {sign: 0, mantissa: 3474716, exponent: 93 });
	let B_0_1_0 = addFloats(A_1_1_0, A_1_1_1);
	let B_0_1_1 = addFloats(B_0_1_0, A_1_1_2);
	let B_0_1_2 = addFloats(B_0_1_1, A_1_1_3);
	let WXb_0_1 = addFloats(B_0_1_2, Float {sign: 1, mantissa: 1107294, exponent: 94 });
	let X_1_1 = relu(WXb_0_1);
	let A_2_0_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 9312850, exponent: 93 });
	let A_2_0_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 5679134, exponent: 93 });
	let B_1_0_0 = addFloats(A_2_0_0, A_2_0_1);
	let WXb_1_0 = addFloats(B_1_0_0, Float {sign: 1, mantissa: 2597790, exponent: 93 });
	let X_2_0 = relu(WXb_1_0);
	let A_2_1_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 1726093, exponent: 94 });
	let A_2_1_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 6160490, exponent: 93 });
	let B_1_1_0 = addFloats(A_2_1_0, A_2_1_1);
	let WXb_1_1 = addFloats(B_1_1_0, Float {sign: 0, mantissa: 1237475, exponent: 94 });
	let X_2_1 = relu(WXb_1_1);
	let A_2_2_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 1332944, exponent: 93 });
	let A_2_2_1 = mulFloats(X_1_1, Float {sign: 0, mantissa: 5227415, exponent: 93 });
	let B_1_2_0 = addFloats(A_2_2_0, A_2_2_1);
	let WXb_1_2 = addFloats(B_1_2_0, Float {sign: 0, mantissa: 5072440, exponent: 93 });
	let X_2_2 = relu(WXb_1_2);
	let A_3_0_0 = mulFloats(X_2_0, Float {sign: 0, mantissa: 1251275, exponent: 94 });
	let A_3_0_1 = mulFloats(X_2_1, Float {sign: 0, mantissa: 9338280, exponent: 93 });
	let A_3_0_2 = mulFloats(X_2_2, Float {sign: 1, mantissa: 5691453, exponent: 93 });
	let B_2_0_0 = addFloats(A_3_0_0, A_3_0_1);
	let B_2_0_1 = addFloats(B_2_0_0, A_3_0_2);
	let WXb_2_0 = addFloats(B_2_0_1, Float {sign: 1, mantissa: 1903210, exponent: 94 });
	let O_3_0 = relu(WXb_2_0);
	let A_3_1_0 = mulFloats(X_2_0, Float {sign: 1, mantissa: 1404632, exponent: 94 });
	let A_3_1_1 = mulFloats(X_2_1, Float {sign: 0, mantissa: 1359964, exponent: 94 });
	let A_3_1_2 = mulFloats(X_2_2, Float {sign: 1, mantissa: 1440878, exponent: 93 });
	let B_2_1_0 = addFloats(A_3_1_0, A_3_1_1);
	let B_2_1_1 = addFloats(B_2_1_0, A_3_1_2);
	let WXb_2_1 = addFloats(B_2_1_1, Float {sign: 0, mantissa: 1078991, exponent: 93 });
	let O_3_1 = relu(WXb_2_1);
	let A_3_2_0 = mulFloats(X_2_0, Float {sign: 0, mantissa: 1641625, exponent: 93 });
	let A_3_2_1 = mulFloats(X_2_1, Float {sign: 1, mantissa: 9258289, exponent: 93 });
	let A_3_2_2 = mulFloats(X_2_2, Float {sign: 0, mantissa: 5135948, exponent: 93 });
	let B_2_2_0 = addFloats(A_3_2_0, A_3_2_1);
	let B_2_2_1 = addFloats(B_2_2_0, A_3_2_2);
	let WXb_2_2 = addFloats(B_2_2_1, Float {sign: 0, mantissa: 1692569, exponent: 94 });
	let O_3_2 = relu(WXb_2_2);

	constrain O_3_0.sign == y_1[0];
	constrain O_3_0.mantissa == y_1[1];
	constrain O_3_0.exponent == y_1[2];
	constrain O_3_1.sign == y_2[0];
	constrain O_3_1.mantissa == y_2[1];
	constrain O_3_1.exponent == y_2[2];
	constrain O_3_2.sign == y_3[0];
	constrain O_3_2.mantissa == y_3[1];
	constrain O_3_2.exponent == y_3[2];
}
// A_1_0_0 = 5100000e94 * 1330295e94 = 6784504e94
// A_1_0_1 = 3500000e94 * 5248411e93 = 1836943e94
// A_1_0_2 = 1400000e94 * -1029153e94 = -1440814e94
// A_1_0_3 = 2000000e93 * -1465364e94 = -2930728e93
// B_0_0_0 = 6784504e94 + 1836943e94 = 8621447e94
// B_0_0_1 = 8621447e94 + -1440814e94 = 7180633e94
// B_0_0_2 = 7180633e94 + -2930728e93 = 6887560e94
// WXb_0_0 = 6887560e94 + 4351563e93 = 7322716e94
// A_1_1_0 = 5100000e94 * 6988025e93 = 3563892e94
// A_1_1_1 = 3500000e94 * -6373806e93 = -2230832e94
// A_1_1_2 = 1400000e94 * 9324125e93 = 1305377e94
// A_1_1_3 = 2000000e93 * 3474716e93 = 6949432e92
// B_0_1_0 = 3563892e94 + -2230832e94 = 1333060e94
// B_0_1_1 = 1333060e94 + 1305377e94 = 2638437e94
// B_0_1_2 = 2638437e94 + 6949432e92 = 2707931e94
// WXb_0_1 = 2707931e94 + -1107294e94 = 1600637e94
// A_2_0_0 = 7322716e94 * 9312850e93 = 6819535e94
// A_2_0_1 = 1600637e94 * -5679134e93 = -9090232e93
// B_1_0_0 = 6819535e94 + -9090232e93 = 5910511e94
// WXb_1_0 = 5910511e94 + -2597790e93 = 5650732e94
// A_2_1_0 = 7322716e94 * 1726093e94 = 1263968e95
// A_2_1_1 = 1600637e94 * -6160490e93 = -9860708e93
// B_1_1_0 = 1263968e95 + -9860708e93 = 1165360e95
// WXb_1_1 = 1165360e95 + 1237475e94 = 1289107e95
// A_2_2_0 = 7322716e94 * 1332944e93 = 9760770e93
// A_2_2_1 = 1600637e94 * 5227415e93 = 8367193e93
// B_1_2_0 = 9760770e93 + 8367193e93 = 1812796e94
// WXb_1_2 = 1812796e94 + 5072440e93 = 2320040e94
// A_3_0_0 = 5650732e94 * 1251275e94 = 7070619e94
// A_3_0_1 = 1289107e95 * 9338280e93 = 1203804e95
// A_3_0_2 = 2320040e94 * -5691453e93 = -1320439e94
// B_2_0_0 = 7070619e94 + 1203804e95 = 1910865e95
// B_2_0_1 = 1910865e95 + -1320439e94 = 1778821e95
// WXb_2_0 = 1778821e95 + -1903210e94 = 1588500e95
// A_3_1_0 = 5650732e94 * -1404632e94 = -7937198e94
// A_3_1_1 = 1289107e95 * 1359964e94 = 1753139e95
// A_3_1_2 = 2320040e94 * -1440878e93 = -3342894e93
// B_2_1_0 = -7937198e94 + 1753139e95 = 9594192e94
// B_2_1_1 = 9594192e94 + -3342894e93 = 9259902e94
// WXb_2_1 = 9259902e94 + 1078991e93 = 9367801e94
// A_3_2_0 = 5650732e94 * 1641625e93 = 9276382e93
// A_3_2_1 = 1289107e95 * -9258289e93 = -1193492e95
// A_3_2_2 = 2320040e94 * 5135948e93 = 1191560e94
// B_2_2_0 = 9276382e93 + -1193492e95 = -1100728e95
// B_2_2_1 = -1100728e95 + 1191560e94 = -9815720e94
// WXb_2_2 = -9815720e94 + 1692569e94 = -8123151e94
// X_0_0 => 5100000e94
// X_0_1 => 3500000e94
// X_0_2 => 1400000e94
// X_0_3 => 2000000e93
// A_1_0_0 => 6784504e94
// A_1_0_1 => 1836943e94
// A_1_0_2 => -1440814e94
// A_1_0_3 => -2930728e93
// B_0_0_0 => 8621447e94
// B_0_0_1 => 7180633e94
// B_0_0_2 => 6887560e94
// WXb_0_0 => 7322716e94
// X_1_0 => 7322716e94
// A_1_1_0 => 3563892e94
// A_1_1_1 => -2230832e94
// A_1_1_2 => 1305377e94
// A_1_1_3 => 6949432e92
// B_0_1_0 => 1333060e94
// B_0_1_1 => 2638437e94
// B_0_1_2 => 2707931e94
// WXb_0_1 => 1600637e94
// X_1_1 => 1600637e94
// A_2_0_0 => 6819535e94
// A_2_0_1 => -9090232e93
// B_1_0_0 => 5910511e94
// WXb_1_0 => 5650732e94
// X_2_0 => 5650732e94
// A_2_1_0 => 1263968e95
// A_2_1_1 => -9860708e93
// B_1_1_0 => 1165360e95
// WXb_1_1 => 1289107e95
// X_2_1 => 1289107e95
// A_2_2_0 => 9760770e93
// A_2_2_1 => 8367193e93
// B_1_2_0 => 1812796e94
// WXb_1_2 => 2320040e94
// X_2_2 => 2320040e94
// A_3_0_0 => 7070619e94
// A_3_0_1 => 1203804e95
// A_3_0_2 => -1320439e94
// B_2_0_0 => 1910865e95
// B_2_0_1 => 1778821e95
// WXb_2_0 => 1588500e95
// O_3_0 => 1588500e95
// A_3_1_0 => -7937198e94
// A_3_1_1 => 1753139e95
// A_3_1_2 => -3342894e93
// B_2_1_0 => 9594192e94
// B_2_1_1 => 9259902e94
// WXb_2_1 => 9367801e94
// O_3_1 => 9367801e94
// A_3_2_0 => 9276382e93
// A_3_2_1 => -1193492e95
// A_3_2_2 => 1191560e94
// B_2_2_0 => -1100728e95
// B_2_2_1 => -9815720e94
// WXb_2_2 => -8123151e94
// O_3_2 => 0e100
