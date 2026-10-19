// SPDX-License-Identifier: Apache-2.0 OR MIT
use super::{add, cheap, mul, sub};
// Fixed schedules: public small-root powers and no value-dependent branches.
#[inline(always)]
pub(super) fn small2<const LAZY: bool, const GS: bool>(x: &mut [u64; 16], _inverse: bool) {
    if GS {
        {
            let a = x[0];
            let b = x[1];
            let t = sub::<LAZY>(a, b);
            x[0] = add::<LAZY>(a, b);
            x[1] = t;
        }
    } else {
        {
            let a = x[0];
            let b = x[1];
            x[0] = add::<LAZY>(a, b);
            x[1] = sub::<LAZY>(a, b);
        }
    }
}
#[inline(always)]
pub(super) fn small4<const LAZY: bool, const GS: bool>(x: &mut [u64; 16], inverse: bool) {
    if GS {
        {
            let a = x[0];
            let b = x[1];
            let t = sub::<LAZY>(a, b);
            x[0] = add::<LAZY>(a, b);
            x[1] = t;
        }
        {
            let a = x[2];
            let b = x[3];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[2] = add::<LAZY>(a, b);
            x[3] = t;
        }
        {
            let a = x[0];
            let b = x[2];
            let t = sub::<LAZY>(a, b);
            x[0] = add::<LAZY>(a, b);
            x[2] = t;
        }
        {
            let a = x[1];
            let b = x[3];
            let t = sub::<LAZY>(a, b);
            x[1] = add::<LAZY>(a, b);
            x[3] = t;
        }
    } else {
        {
            let a = x[0];
            let b = x[2];
            x[0] = add::<LAZY>(a, b);
            x[2] = sub::<LAZY>(a, b);
        }
        {
            let a = x[1];
            let b = x[3];
            x[1] = add::<LAZY>(a, b);
            x[3] = sub::<LAZY>(a, b);
        }
        {
            let a = x[0];
            let b = x[1];
            x[0] = add::<LAZY>(a, b);
            x[1] = sub::<LAZY>(a, b);
        }
        {
            let a = x[2];
            let b = x[3];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[2] = add::<LAZY>(a, b);
            x[3] = sub::<LAZY>(a, b);
        }
    }
}
#[inline(always)]
pub(super) fn small8<const LAZY: bool, const GS: bool>(x: &mut [u64; 16], inverse: bool) {
    if GS {
        {
            let a = x[0];
            let b = x[1];
            let t = sub::<LAZY>(a, b);
            x[0] = add::<LAZY>(a, b);
            x[1] = t;
        }
        {
            let a = x[2];
            let b = x[3];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[2] = add::<LAZY>(a, b);
            x[3] = t;
        }
        {
            let a = x[4];
            let b = x[5];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 72 } else { 120 });
            x[4] = add::<LAZY>(a, b);
            x[5] = t;
        }
        {
            let a = x[6];
            let b = x[7];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 24 } else { 168 });
            x[6] = add::<LAZY>(a, b);
            x[7] = t;
        }
        {
            let a = x[0];
            let b = x[2];
            let t = sub::<LAZY>(a, b);
            x[0] = add::<LAZY>(a, b);
            x[2] = t;
        }
        {
            let a = x[1];
            let b = x[3];
            let t = sub::<LAZY>(a, b);
            x[1] = add::<LAZY>(a, b);
            x[3] = t;
        }
        {
            let a = x[4];
            let b = x[6];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[4] = add::<LAZY>(a, b);
            x[6] = t;
        }
        {
            let a = x[5];
            let b = x[7];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[5] = add::<LAZY>(a, b);
            x[7] = t;
        }
        {
            let a = x[0];
            let b = x[4];
            let t = sub::<LAZY>(a, b);
            x[0] = add::<LAZY>(a, b);
            x[4] = t;
        }
        {
            let a = x[1];
            let b = x[5];
            let t = sub::<LAZY>(a, b);
            x[1] = add::<LAZY>(a, b);
            x[5] = t;
        }
        {
            let a = x[2];
            let b = x[6];
            let t = sub::<LAZY>(a, b);
            x[2] = add::<LAZY>(a, b);
            x[6] = t;
        }
        {
            let a = x[3];
            let b = x[7];
            let t = sub::<LAZY>(a, b);
            x[3] = add::<LAZY>(a, b);
            x[7] = t;
        }
    } else {
        {
            let a = x[0];
            let b = x[4];
            x[0] = add::<LAZY>(a, b);
            x[4] = sub::<LAZY>(a, b);
        }
        {
            let a = x[1];
            let b = x[5];
            x[1] = add::<LAZY>(a, b);
            x[5] = sub::<LAZY>(a, b);
        }
        {
            let a = x[2];
            let b = x[6];
            x[2] = add::<LAZY>(a, b);
            x[6] = sub::<LAZY>(a, b);
        }
        {
            let a = x[3];
            let b = x[7];
            x[3] = add::<LAZY>(a, b);
            x[7] = sub::<LAZY>(a, b);
        }
        {
            let a = x[0];
            let b = x[2];
            x[0] = add::<LAZY>(a, b);
            x[2] = sub::<LAZY>(a, b);
        }
        {
            let a = x[1];
            let b = x[3];
            x[1] = add::<LAZY>(a, b);
            x[3] = sub::<LAZY>(a, b);
        }
        {
            let a = x[4];
            let b = x[6];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[4] = add::<LAZY>(a, b);
            x[6] = sub::<LAZY>(a, b);
        }
        {
            let a = x[5];
            let b = x[7];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[5] = add::<LAZY>(a, b);
            x[7] = sub::<LAZY>(a, b);
        }
        {
            let a = x[0];
            let b = x[1];
            x[0] = add::<LAZY>(a, b);
            x[1] = sub::<LAZY>(a, b);
        }
        {
            let a = x[2];
            let b = x[3];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[2] = add::<LAZY>(a, b);
            x[3] = sub::<LAZY>(a, b);
        }
        {
            let a = x[4];
            let b = x[5];
            let b = cheap::<LAZY>(b, if inverse { 72 } else { 120 });
            x[4] = add::<LAZY>(a, b);
            x[5] = sub::<LAZY>(a, b);
        }
        {
            let a = x[6];
            let b = x[7];
            let b = cheap::<LAZY>(b, if inverse { 24 } else { 168 });
            x[6] = add::<LAZY>(a, b);
            x[7] = sub::<LAZY>(a, b);
        }
    }
}
#[inline(always)]
pub(super) fn small16<const LAZY: bool, const GS: bool>(x: &mut [u64; 16], inverse: bool) {
    if GS {
        {
            let a = x[0];
            let b = x[1];
            let t = sub::<LAZY>(a, b);
            x[0] = add::<LAZY>(a, b);
            x[1] = t;
        }
        {
            let a = x[2];
            let b = x[3];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[2] = add::<LAZY>(a, b);
            x[3] = t;
        }
        {
            let a = x[4];
            let b = x[5];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 72 } else { 120 });
            x[4] = add::<LAZY>(a, b);
            x[5] = t;
        }
        {
            let a = x[6];
            let b = x[7];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 24 } else { 168 });
            x[6] = add::<LAZY>(a, b);
            x[7] = t;
        }
        {
            let a = x[8];
            let b = x[9];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 36 } else { 156 });
            x[8] = add::<LAZY>(a, b);
            x[9] = t;
        }
        {
            let a = x[10];
            let b = x[11];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 180 } else { 12 });
            x[10] = add::<LAZY>(a, b);
            x[11] = t;
        }
        {
            let a = x[12];
            let b = x[13];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 108 } else { 84 });
            x[12] = add::<LAZY>(a, b);
            x[13] = t;
        }
        {
            let a = x[14];
            let b = x[15];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 60 } else { 132 });
            x[14] = add::<LAZY>(a, b);
            x[15] = t;
        }
        {
            let a = x[0];
            let b = x[2];
            let t = sub::<LAZY>(a, b);
            x[0] = add::<LAZY>(a, b);
            x[2] = t;
        }
        {
            let a = x[1];
            let b = x[3];
            let t = sub::<LAZY>(a, b);
            x[1] = add::<LAZY>(a, b);
            x[3] = t;
        }
        {
            let a = x[4];
            let b = x[6];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[4] = add::<LAZY>(a, b);
            x[6] = t;
        }
        {
            let a = x[5];
            let b = x[7];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[5] = add::<LAZY>(a, b);
            x[7] = t;
        }
        {
            let a = x[8];
            let b = x[10];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 72 } else { 120 });
            x[8] = add::<LAZY>(a, b);
            x[10] = t;
        }
        {
            let a = x[9];
            let b = x[11];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 72 } else { 120 });
            x[9] = add::<LAZY>(a, b);
            x[11] = t;
        }
        {
            let a = x[12];
            let b = x[14];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 24 } else { 168 });
            x[12] = add::<LAZY>(a, b);
            x[14] = t;
        }
        {
            let a = x[13];
            let b = x[15];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 24 } else { 168 });
            x[13] = add::<LAZY>(a, b);
            x[15] = t;
        }
        {
            let a = x[0];
            let b = x[4];
            let t = sub::<LAZY>(a, b);
            x[0] = add::<LAZY>(a, b);
            x[4] = t;
        }
        {
            let a = x[1];
            let b = x[5];
            let t = sub::<LAZY>(a, b);
            x[1] = add::<LAZY>(a, b);
            x[5] = t;
        }
        {
            let a = x[2];
            let b = x[6];
            let t = sub::<LAZY>(a, b);
            x[2] = add::<LAZY>(a, b);
            x[6] = t;
        }
        {
            let a = x[3];
            let b = x[7];
            let t = sub::<LAZY>(a, b);
            x[3] = add::<LAZY>(a, b);
            x[7] = t;
        }
        {
            let a = x[8];
            let b = x[12];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[8] = add::<LAZY>(a, b);
            x[12] = t;
        }
        {
            let a = x[9];
            let b = x[13];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[9] = add::<LAZY>(a, b);
            x[13] = t;
        }
        {
            let a = x[10];
            let b = x[14];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[10] = add::<LAZY>(a, b);
            x[14] = t;
        }
        {
            let a = x[11];
            let b = x[15];
            let t = sub::<LAZY>(a, b);
            let t = cheap::<LAZY>(t, if inverse { 144 } else { 48 });
            x[11] = add::<LAZY>(a, b);
            x[15] = t;
        }
        {
            let a = x[0];
            let b = x[8];
            let t = sub::<LAZY>(a, b);
            x[0] = add::<LAZY>(a, b);
            x[8] = t;
        }
        {
            let a = x[1];
            let b = x[9];
            let t = sub::<LAZY>(a, b);
            x[1] = add::<LAZY>(a, b);
            x[9] = t;
        }
        {
            let a = x[2];
            let b = x[10];
            let t = sub::<LAZY>(a, b);
            x[2] = add::<LAZY>(a, b);
            x[10] = t;
        }
        {
            let a = x[3];
            let b = x[11];
            let t = sub::<LAZY>(a, b);
            x[3] = add::<LAZY>(a, b);
            x[11] = t;
        }
        {
            let a = x[4];
            let b = x[12];
            let t = sub::<LAZY>(a, b);
            x[4] = add::<LAZY>(a, b);
            x[12] = t;
        }
        {
            let a = x[5];
            let b = x[13];
            let t = sub::<LAZY>(a, b);
            x[5] = add::<LAZY>(a, b);
            x[13] = t;
        }
        {
            let a = x[6];
            let b = x[14];
            let t = sub::<LAZY>(a, b);
            x[6] = add::<LAZY>(a, b);
            x[14] = t;
        }
        {
            let a = x[7];
            let b = x[15];
            let t = sub::<LAZY>(a, b);
            x[7] = add::<LAZY>(a, b);
            x[15] = t;
        }
    } else {
        {
            let a = x[0];
            let b = x[8];
            x[0] = add::<LAZY>(a, b);
            x[8] = sub::<LAZY>(a, b);
        }
        {
            let a = x[1];
            let b = x[9];
            x[1] = add::<LAZY>(a, b);
            x[9] = sub::<LAZY>(a, b);
        }
        {
            let a = x[2];
            let b = x[10];
            x[2] = add::<LAZY>(a, b);
            x[10] = sub::<LAZY>(a, b);
        }
        {
            let a = x[3];
            let b = x[11];
            x[3] = add::<LAZY>(a, b);
            x[11] = sub::<LAZY>(a, b);
        }
        {
            let a = x[4];
            let b = x[12];
            x[4] = add::<LAZY>(a, b);
            x[12] = sub::<LAZY>(a, b);
        }
        {
            let a = x[5];
            let b = x[13];
            x[5] = add::<LAZY>(a, b);
            x[13] = sub::<LAZY>(a, b);
        }
        {
            let a = x[6];
            let b = x[14];
            x[6] = add::<LAZY>(a, b);
            x[14] = sub::<LAZY>(a, b);
        }
        {
            let a = x[7];
            let b = x[15];
            x[7] = add::<LAZY>(a, b);
            x[15] = sub::<LAZY>(a, b);
        }
        {
            let a = x[0];
            let b = x[4];
            x[0] = add::<LAZY>(a, b);
            x[4] = sub::<LAZY>(a, b);
        }
        {
            let a = x[1];
            let b = x[5];
            x[1] = add::<LAZY>(a, b);
            x[5] = sub::<LAZY>(a, b);
        }
        {
            let a = x[2];
            let b = x[6];
            x[2] = add::<LAZY>(a, b);
            x[6] = sub::<LAZY>(a, b);
        }
        {
            let a = x[3];
            let b = x[7];
            x[3] = add::<LAZY>(a, b);
            x[7] = sub::<LAZY>(a, b);
        }
        {
            let a = x[8];
            let b = x[12];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[8] = add::<LAZY>(a, b);
            x[12] = sub::<LAZY>(a, b);
        }
        {
            let a = x[9];
            let b = x[13];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[9] = add::<LAZY>(a, b);
            x[13] = sub::<LAZY>(a, b);
        }
        {
            let a = x[10];
            let b = x[14];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[10] = add::<LAZY>(a, b);
            x[14] = sub::<LAZY>(a, b);
        }
        {
            let a = x[11];
            let b = x[15];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[11] = add::<LAZY>(a, b);
            x[15] = sub::<LAZY>(a, b);
        }
        {
            let a = x[0];
            let b = x[2];
            x[0] = add::<LAZY>(a, b);
            x[2] = sub::<LAZY>(a, b);
        }
        {
            let a = x[1];
            let b = x[3];
            x[1] = add::<LAZY>(a, b);
            x[3] = sub::<LAZY>(a, b);
        }
        {
            let a = x[4];
            let b = x[6];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[4] = add::<LAZY>(a, b);
            x[6] = sub::<LAZY>(a, b);
        }
        {
            let a = x[5];
            let b = x[7];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[5] = add::<LAZY>(a, b);
            x[7] = sub::<LAZY>(a, b);
        }
        {
            let a = x[8];
            let b = x[10];
            let b = cheap::<LAZY>(b, if inverse { 72 } else { 120 });
            x[8] = add::<LAZY>(a, b);
            x[10] = sub::<LAZY>(a, b);
        }
        {
            let a = x[9];
            let b = x[11];
            let b = cheap::<LAZY>(b, if inverse { 72 } else { 120 });
            x[9] = add::<LAZY>(a, b);
            x[11] = sub::<LAZY>(a, b);
        }
        {
            let a = x[12];
            let b = x[14];
            let b = cheap::<LAZY>(b, if inverse { 24 } else { 168 });
            x[12] = add::<LAZY>(a, b);
            x[14] = sub::<LAZY>(a, b);
        }
        {
            let a = x[13];
            let b = x[15];
            let b = cheap::<LAZY>(b, if inverse { 24 } else { 168 });
            x[13] = add::<LAZY>(a, b);
            x[15] = sub::<LAZY>(a, b);
        }
        {
            let a = x[0];
            let b = x[1];
            x[0] = add::<LAZY>(a, b);
            x[1] = sub::<LAZY>(a, b);
        }
        {
            let a = x[2];
            let b = x[3];
            let b = cheap::<LAZY>(b, if inverse { 144 } else { 48 });
            x[2] = add::<LAZY>(a, b);
            x[3] = sub::<LAZY>(a, b);
        }
        {
            let a = x[4];
            let b = x[5];
            let b = cheap::<LAZY>(b, if inverse { 72 } else { 120 });
            x[4] = add::<LAZY>(a, b);
            x[5] = sub::<LAZY>(a, b);
        }
        {
            let a = x[6];
            let b = x[7];
            let b = cheap::<LAZY>(b, if inverse { 24 } else { 168 });
            x[6] = add::<LAZY>(a, b);
            x[7] = sub::<LAZY>(a, b);
        }
        {
            let a = x[8];
            let b = x[9];
            let b = cheap::<LAZY>(b, if inverse { 36 } else { 156 });
            x[8] = add::<LAZY>(a, b);
            x[9] = sub::<LAZY>(a, b);
        }
        {
            let a = x[10];
            let b = x[11];
            let b = cheap::<LAZY>(b, if inverse { 180 } else { 12 });
            x[10] = add::<LAZY>(a, b);
            x[11] = sub::<LAZY>(a, b);
        }
        {
            let a = x[12];
            let b = x[13];
            let b = cheap::<LAZY>(b, if inverse { 108 } else { 84 });
            x[12] = add::<LAZY>(a, b);
            x[13] = sub::<LAZY>(a, b);
        }
        {
            let a = x[14];
            let b = x[15];
            let b = cheap::<LAZY>(b, if inverse { 60 } else { 132 });
            x[14] = add::<LAZY>(a, b);
            x[15] = sub::<LAZY>(a, b);
        }
    }
}
#[inline(never)]
pub(super) fn lazy_radix16_anchor(x: &mut [u64; 16], powers: &[u64; 15]) {
    x[1] = mul::<true>(x[1], powers[0]);
    x[2] = mul::<true>(x[2], powers[1]);
    x[3] = mul::<true>(x[3], powers[2]);
    x[4] = mul::<true>(x[4], powers[3]);
    x[5] = mul::<true>(x[5], powers[4]);
    x[6] = mul::<true>(x[6], powers[5]);
    x[7] = mul::<true>(x[7], powers[6]);
    x[8] = mul::<true>(x[8], powers[7]);
    x[9] = mul::<true>(x[9], powers[8]);
    x[10] = mul::<true>(x[10], powers[9]);
    x[11] = mul::<true>(x[11], powers[10]);
    x[12] = mul::<true>(x[12], powers[11]);
    x[13] = mul::<true>(x[13], powers[12]);
    x[14] = mul::<true>(x[14], powers[13]);
    x[15] = mul::<true>(x[15], powers[14]);
    small16::<true, false>(x, false);
}
#[inline(never)]
pub(super) fn radix2_anchor(x: &mut [u64; 16], tw: &[u64; 15]) {
    {
        let a = x[0];
        let b = mul::<false>(x[8], tw[0]);
        x[0] = add::<false>(a, b);
        x[8] = sub::<false>(a, b);
    }
    {
        let a = x[1];
        let b = mul::<false>(x[9], tw[0]);
        x[1] = add::<false>(a, b);
        x[9] = sub::<false>(a, b);
    }
    {
        let a = x[2];
        let b = mul::<false>(x[10], tw[0]);
        x[2] = add::<false>(a, b);
        x[10] = sub::<false>(a, b);
    }
    {
        let a = x[3];
        let b = mul::<false>(x[11], tw[0]);
        x[3] = add::<false>(a, b);
        x[11] = sub::<false>(a, b);
    }
    {
        let a = x[4];
        let b = mul::<false>(x[12], tw[0]);
        x[4] = add::<false>(a, b);
        x[12] = sub::<false>(a, b);
    }
    {
        let a = x[5];
        let b = mul::<false>(x[13], tw[0]);
        x[5] = add::<false>(a, b);
        x[13] = sub::<false>(a, b);
    }
    {
        let a = x[6];
        let b = mul::<false>(x[14], tw[0]);
        x[6] = add::<false>(a, b);
        x[14] = sub::<false>(a, b);
    }
    {
        let a = x[7];
        let b = mul::<false>(x[15], tw[0]);
        x[7] = add::<false>(a, b);
        x[15] = sub::<false>(a, b);
    }
    {
        let a = x[0];
        let b = mul::<false>(x[4], tw[1]);
        x[0] = add::<false>(a, b);
        x[4] = sub::<false>(a, b);
    }
    {
        let a = x[1];
        let b = mul::<false>(x[5], tw[1]);
        x[1] = add::<false>(a, b);
        x[5] = sub::<false>(a, b);
    }
    {
        let a = x[2];
        let b = mul::<false>(x[6], tw[1]);
        x[2] = add::<false>(a, b);
        x[6] = sub::<false>(a, b);
    }
    {
        let a = x[3];
        let b = mul::<false>(x[7], tw[1]);
        x[3] = add::<false>(a, b);
        x[7] = sub::<false>(a, b);
    }
    {
        let a = x[8];
        let b = mul::<false>(x[12], tw[2]);
        x[8] = add::<false>(a, b);
        x[12] = sub::<false>(a, b);
    }
    {
        let a = x[9];
        let b = mul::<false>(x[13], tw[2]);
        x[9] = add::<false>(a, b);
        x[13] = sub::<false>(a, b);
    }
    {
        let a = x[10];
        let b = mul::<false>(x[14], tw[2]);
        x[10] = add::<false>(a, b);
        x[14] = sub::<false>(a, b);
    }
    {
        let a = x[11];
        let b = mul::<false>(x[15], tw[2]);
        x[11] = add::<false>(a, b);
        x[15] = sub::<false>(a, b);
    }
    {
        let a = x[0];
        let b = mul::<false>(x[2], tw[3]);
        x[0] = add::<false>(a, b);
        x[2] = sub::<false>(a, b);
    }
    {
        let a = x[1];
        let b = mul::<false>(x[3], tw[3]);
        x[1] = add::<false>(a, b);
        x[3] = sub::<false>(a, b);
    }
    {
        let a = x[4];
        let b = mul::<false>(x[6], tw[4]);
        x[4] = add::<false>(a, b);
        x[6] = sub::<false>(a, b);
    }
    {
        let a = x[5];
        let b = mul::<false>(x[7], tw[4]);
        x[5] = add::<false>(a, b);
        x[7] = sub::<false>(a, b);
    }
    {
        let a = x[8];
        let b = mul::<false>(x[10], tw[5]);
        x[8] = add::<false>(a, b);
        x[10] = sub::<false>(a, b);
    }
    {
        let a = x[9];
        let b = mul::<false>(x[11], tw[5]);
        x[9] = add::<false>(a, b);
        x[11] = sub::<false>(a, b);
    }
    {
        let a = x[12];
        let b = mul::<false>(x[14], tw[6]);
        x[12] = add::<false>(a, b);
        x[14] = sub::<false>(a, b);
    }
    {
        let a = x[13];
        let b = mul::<false>(x[15], tw[6]);
        x[13] = add::<false>(a, b);
        x[15] = sub::<false>(a, b);
    }
    {
        let a = x[0];
        let b = mul::<false>(x[1], tw[7]);
        x[0] = add::<false>(a, b);
        x[1] = sub::<false>(a, b);
    }
    {
        let a = x[2];
        let b = mul::<false>(x[3], tw[8]);
        x[2] = add::<false>(a, b);
        x[3] = sub::<false>(a, b);
    }
    {
        let a = x[4];
        let b = mul::<false>(x[5], tw[9]);
        x[4] = add::<false>(a, b);
        x[5] = sub::<false>(a, b);
    }
    {
        let a = x[6];
        let b = mul::<false>(x[7], tw[10]);
        x[6] = add::<false>(a, b);
        x[7] = sub::<false>(a, b);
    }
    {
        let a = x[8];
        let b = mul::<false>(x[9], tw[11]);
        x[8] = add::<false>(a, b);
        x[9] = sub::<false>(a, b);
    }
    {
        let a = x[10];
        let b = mul::<false>(x[11], tw[12]);
        x[10] = add::<false>(a, b);
        x[11] = sub::<false>(a, b);
    }
    {
        let a = x[12];
        let b = mul::<false>(x[13], tw[13]);
        x[12] = add::<false>(a, b);
        x[13] = sub::<false>(a, b);
    }
    {
        let a = x[14];
        let b = mul::<false>(x[15], tw[14]);
        x[14] = add::<false>(a, b);
        x[15] = sub::<false>(a, b);
    }
}
