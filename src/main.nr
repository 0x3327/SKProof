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
	let A_1_0_0 = mulFloats(X_0_0, Float {sign: 0, mantissa: 1336663, exponent: 93 });
	let A_1_0_1 = mulFloats(X_0_1, Float {sign: 0, mantissa: 7586549, exponent: 93 });
	let A_1_0_2 = mulFloats(X_0_2, Float {sign: 0, mantissa: 1043080, exponent: 94 });
	let A_1_0_3 = mulFloats(X_0_3, Float {sign: 0, mantissa: 1170603, exponent: 94 });
	let B_0_0_0 = addFloats(A_1_0_0, A_1_0_1);
	let B_0_0_1 = addFloats(B_0_0_0, A_1_0_2);
	let B_0_0_2 = addFloats(B_0_0_1, A_1_0_3);
	let WXb_0_0 = addFloats(B_0_0_2, Float {sign: 0, mantissa: 4078493, exponent: 93 });
	let X_1_0 = relu(WXb_0_0);
	let A_1_1_0 = mulFloats(X_0_0, Float {sign: 0, mantissa: 7829273, exponent: 93 });
	let A_1_1_1 = mulFloats(X_0_1, Float {sign: 0, mantissa: 1365790, exponent: 94 });
	let A_1_1_2 = mulFloats(X_0_2, Float {sign: 1, mantissa: 1386341, exponent: 94 });
	let A_1_1_3 = mulFloats(X_0_3, Float {sign: 1, mantissa: 1053194, exponent: 94 });
	let B_0_1_0 = addFloats(A_1_1_0, A_1_1_1);
	let B_0_1_1 = addFloats(B_0_1_0, A_1_1_2);
	let B_0_1_2 = addFloats(B_0_1_1, A_1_1_3);
	let WXb_0_1 = addFloats(B_0_1_2, Float {sign: 0, mantissa: 2466248, exponent: 94 });
	let X_1_1 = relu(WXb_0_1);
	let A_2_0_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 1152279, exponent: 94 });
	let A_2_0_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 9092448, exponent: 93 });
	let B_1_0_0 = addFloats(A_2_0_0, A_2_0_1);
	let WXb_1_0 = addFloats(B_1_0_0, Float {sign: 0, mantissa: 1596867, exponent: 93 });
	let X_2_0 = relu(WXb_1_0);
	let A_2_1_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 1101590, exponent: 92 });
	let A_2_1_1 = mulFloats(X_1_1, Float {sign: 0, mantissa: 1627253, exponent: 94 });
	let B_1_1_0 = addFloats(A_2_1_0, A_2_1_1);
	let WXb_1_1 = addFloats(B_1_1_0, Float {sign: 0, mantissa: 3013073, exponent: 93 });
	let X_2_1 = relu(WXb_1_1);
	let A_2_2_0 = mulFloats(X_1_0, Float {sign: 1, mantissa: 9135291, exponent: 88 });
	let A_2_2_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 2661389, exponent: 91 });
	let B_1_2_0 = addFloats(A_2_2_0, A_2_2_1);
	let WXb_1_2 = addFloats(B_1_2_0, Float {sign: 1, mantissa: 3054311, exponent: 93 });
	let X_2_2 = relu(WXb_1_2);
	let A_3_0_0 = mulFloats(X_2_0, Float {sign: 1, mantissa: 6605394, exponent: 93 });
	let A_3_0_1 = mulFloats(X_2_1, Float {sign: 0, mantissa: 3677529, exponent: 93 });
	let A_3_0_2 = mulFloats(X_2_2, Float {sign: 1, mantissa: 14324, exponent: 86 });
	let B_2_0_0 = addFloats(A_3_0_0, A_3_0_1);
	let B_2_0_1 = addFloats(B_2_0_0, A_3_0_2);
	let WXb_2_0 = addFloats(B_2_0_1, Float {sign: 0, mantissa: 1129286, exponent: 93 });
	let O_3_0 = relu(WXb_2_0);
	let A_3_1_0 = mulFloats(X_2_0, Float {sign: 0, mantissa: 4068012, exponent: 93 });
	let A_3_1_1 = mulFloats(X_2_1, Float {sign: 0, mantissa: 3445262, exponent: 92 });
	let A_3_1_2 = mulFloats(X_2_2, Float {sign: 0, mantissa: 3444056, exponent: 90 });
	let B_2_1_0 = addFloats(A_3_1_0, A_3_1_1);
	let B_2_1_1 = addFloats(B_2_1_0, A_3_1_2);
	let WXb_2_1 = addFloats(B_2_1_1, Float {sign: 1, mantissa: 4435015, exponent: 92 });
	let O_3_1 = relu(WXb_2_1);
	let A_3_2_0 = mulFloats(X_2_0, Float {sign: 0, mantissa: 1092163, exponent: 94 });
	let A_3_2_1 = mulFloats(X_2_1, Float {sign: 1, mantissa: 1226450, exponent: 94 });
	let A_3_2_2 = mulFloats(X_2_2, Float {sign: 0, mantissa: 0, exponent: 100 });
	let B_2_2_0 = addFloats(A_3_2_0, A_3_2_1);
	let B_2_2_1 = addFloats(B_2_2_0, A_3_2_2);
	let WXb_2_2 = addFloats(B_2_2_1, Float {sign: 1, mantissa: 5861191, exponent: 93 });
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
// A_1_0_0 = 5100000e94 * 1336663e93 = 6816981e93
// A_1_0_1 = 3500000e94 * 7586549e93 = 2655292e94
// A_1_0_2 = 1400000e94 * 1043080e94 = 1460312e94
// A_1_0_3 = 2000000e93 * 1170603e94 = 2341206e93
// B_0_0_0 = 6816981e93 + 2655292e94 = 3336990e94
// B_0_0_1 = 3336990e94 + 1460312e94 = 4797302e94
// B_0_0_2 = 4797302e94 + 2341206e93 = 5031422e94
// WXb_0_0 = 5031422e94 + 4078493e93 = 5439271e94
// A_1_1_0 = 5100000e94 * 7829273e93 = 3992929e94
// A_1_1_1 = 3500000e94 * 1365790e94 = 4780265e94
// A_1_1_2 = 1400000e94 * -1386341e94 = -1940877e94
// A_1_1_3 = 2000000e93 * -1053194e94 = -2106388e93
// B_0_1_0 = 3992929e94 + 4780265e94 = 8773194e94
// B_0_1_1 = 8773194e94 + -1940877e94 = 6832317e94
// B_0_1_2 = 6832317e94 + -2106388e93 = 6621678e94
// WXb_0_1 = 6621678e94 + 2466248e94 = 9087926e94
// A_2_0_0 = 5439271e94 * 1152279e94 = 6267557e94
// A_2_0_1 = 9087926e94 * -9092448e93 = -8263149e94
// B_1_0_0 = 6267557e94 + -8263149e94 = -1995592e94
// WXb_1_0 = -1995592e94 + 1596867e93 = -1835905e94
// A_2_1_0 = 5439271e94 * 1101590e92 = 5991846e92
// A_2_1_1 = 9087926e94 * 1627253e94 = 1478835e95
// B_1_1_0 = 5991846e92 + 1478835e95 = 1484826e95
// WXb_1_1 = 1484826e95 + 3013073e93 = 1514956e95
// A_2_2_0 = 5439271e94 * -9135291e88 = -4968932e89
// A_2_2_1 = 9087926e94 * -2661389e91 = -2418650e92
// B_1_2_0 = -4968932e89 + -2418650e92 = -2423618e92
// WXb_1_2 = -2423618e92 + -3054311e93 = -3296672e93
// A_3_0_0 = 0e100 * -6605394e93 = 0e100
// A_3_0_1 = 1514956e95 * 3677529e93 = 5571294e94
// A_3_0_2 = 0e100 * -14324e86 = 0e100
// B_2_0_0 = 0e100 + 5571294e94 = 5571294e94
// B_2_0_1 = 5571294e94 + 0e100 = 5571294e94
// WXb_2_0 = 5571294e94 + 1129286e93 = 5684222e94
// A_3_1_0 = 0e100 * 4068012e93 = 0e100
// A_3_1_1 = 1514956e95 * 3445262e92 = 5219420e93
// A_3_1_2 = 0e100 * 3444056e90 = 0e100
// B_2_1_0 = 0e100 + 5219420e93 = 5219420e93
// B_2_1_1 = 5219420e93 + 0e100 = 5219420e93
// WXb_2_1 = 5219420e93 + -4435015e92 = 4775918e93
// A_3_2_0 = 0e100 * 1092163e94 = 0e100
// A_3_2_1 = 1514956e95 * -1226450e94 = -1858017e95
// A_3_2_2 = 0e100 * 0e100 = 0e100
// B_2_2_0 = 0e100 + -1858017e95 = -1858017e95
// B_2_2_1 = -1858017e95 + 0e100 = -1858017e95
// WXb_2_2 = -1858017e95 + -5861191e93 = -1916628e95
// X_0_0 => 5100000e94
// X_0_1 => 3500000e94
// X_0_2 => 1400000e94
// X_0_3 => 2000000e93
// A_1_0_0 => 6816981e93
// A_1_0_1 => 2655292e94
// A_1_0_2 => 1460312e94
// A_1_0_3 => 2341206e93
// B_0_0_0 => 3336990e94
// B_0_0_1 => 4797302e94
// B_0_0_2 => 5031422e94
// WXb_0_0 => 5439271e94
// X_1_0 => 5439271e94
// A_1_1_0 => 3992929e94
// A_1_1_1 => 4780265e94
// A_1_1_2 => -1940877e94
// A_1_1_3 => -2106388e93
// B_0_1_0 => 8773194e94
// B_0_1_1 => 6832317e94
// B_0_1_2 => 6621678e94
// WXb_0_1 => 9087926e94
// X_1_1 => 9087926e94
// A_2_0_0 => 6267557e94
// A_2_0_1 => -8263149e94
// B_1_0_0 => -1995592e94
// WXb_1_0 => -1835905e94
// X_2_0 => 0e100
// A_2_1_0 => 5991846e92
// A_2_1_1 => 1478835e95
// B_1_1_0 => 1484826e95
// WXb_1_1 => 1514956e95
// X_2_1 => 1514956e95
// A_2_2_0 => -4968932e89
// A_2_2_1 => -2418650e92
// B_1_2_0 => -2423618e92
// WXb_1_2 => -3296672e93
// X_2_2 => 0e100
// A_3_0_0 => 0e100
// A_3_0_1 => 5571294e94
// A_3_0_2 => 0e100
// B_2_0_0 => 5571294e94
// B_2_0_1 => 5571294e94
// WXb_2_0 => 5684222e94
// O_3_0 => 5684222e94
// A_3_1_0 => 0e100
// A_3_1_1 => 5219420e93
// A_3_1_2 => 0e100
// B_2_1_0 => 5219420e93
// B_2_1_1 => 5219420e93
// WXb_2_1 => 4775918e93
// O_3_1 => 4775918e93
// A_3_2_0 => 0e100
// A_3_2_1 => -1858017e95
// A_3_2_2 => 0e100
// B_2_2_0 => -1858017e95
// B_2_2_1 => -1858017e95
// WXb_2_2 => -1916628e95
// O_3_2 => 0e100
