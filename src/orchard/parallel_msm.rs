//! Signed-Booth fallback for public Pallas batch-verification terms.
//!
//! Adapted from zakura-core/common's `crates/pasta_curves/src/arithmetic/multiexp.rs`
//! (<https://github.com/zakura-core/common/blob/084ce07f5488cebc580c995bdb7dbf21935f11ff/crates/pasta_curves/src/arithmetic/multiexp.rs>);
//! its Bucket and summation-by-parts originate in zcash/halo2's
//! `halo2_proofs/src/arithmetic.rs` (<https://github.com/zcash/halo2/blob/7465edf9110aace00b8c24b96270e1b820fd39fb/halo2_proofs/src/arithmetic.rs>).
//! MIT OR Apache-2.0, Copyright 2026 The Zakura Contributors.

use alloc::{vec, vec::Vec};

use group::{ff::PrimeField, Group};
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;
use pasta_curves::pallas;

#[derive(Clone, Copy)]
enum Bucket {
    Empty,
    Affine(pallas::Affine),
    Projective(pallas::Point),
}

impl Bucket {
    fn add(&mut self, point: pallas::Affine) {
        *self = match *self {
            Self::Empty => Self::Affine(point),
            Self::Affine(first) => Self::Projective(first + point),
            Self::Projective(mut sum) => {
                sum += point;
                Self::Projective(sum)
            }
        };
    }

    fn sum_with(self, sum: pallas::Point) -> pallas::Point {
        match self {
            Self::Empty => sum,
            Self::Affine(point) => sum + point,
            Self::Projective(point) => sum + point,
        }
    }
}

fn window_bits(terms: usize) -> usize {
    // Floors of e^k, k = 4..=12. Cap the window at 12 to bound bucket memory.
    const THRESHOLDS: [usize; 9] = [54, 148, 403, 1_096, 2_980, 8_103, 22_026, 59_874, 162_754];
    if terms < 4 {
        1
    } else if terms < 32 {
        3
    } else {
        4 + THRESHOLDS
            .iter()
            .position(|bound| terms <= *bound)
            .unwrap_or(THRESHOLDS.len())
    }
}

fn digit(bytes: &[u8], bits: usize, window: usize) -> (usize, bool) {
    let start = window * bits;
    let byte_start = start / 8;
    let mut encoded = [0; 8];
    if let Some(tail) = bytes.get(byte_start..) {
        for (slot, byte) in encoded.iter_mut().zip(tail) {
            *slot = *byte;
        }
    }
    let radix = 1usize << bits;
    let value = ((u64::from_le_bytes(encoded) >> (start % 8)) as usize) & (radix - 1);
    let overlap = if start == 0 {
        0
    } else {
        let bit = start - 1;
        bytes
            .get(bit / 8)
            .map_or(0, |byte| usize::from((byte >> (bit % 8)) & 1))
    };
    if value < radix / 2 {
        (value + overlap, false)
    } else {
        let magnitude = radix - value - overlap;
        (magnitude, magnitude != 0)
    }
}

/// Evaluates one independent window's buckets in its own worker when useful.
pub(super) fn multiexp(scalars: &[pallas::Scalar], bases: &[pallas::Affine]) -> pallas::Point {
    assert_eq!(scalars.len(), bases.len());
    if scalars.is_empty() {
        return pallas::Point::identity();
    }
    let bits = window_bits(scalars.len());
    let representations: Vec<_> = scalars.iter().map(PrimeField::to_repr).collect();
    let windows = 256 / bits + 1;
    let window_sum = |window| {
        let mut buckets = vec![Bucket::Empty; 1 << (bits - 1)];
        for (scalar, point) in representations.iter().zip(bases) {
            let (magnitude, negative) = digit(scalar.as_ref(), bits, window);
            if magnitude != 0 {
                buckets[magnitude - 1].add(if negative { -*point } else { *point });
            }
        }
        let (mut partial, mut sum) = (pallas::Point::identity(), pallas::Point::identity());
        for bucket in buckets.into_iter().rev() {
            sum = bucket.sum_with(sum);
            partial += sum;
        }
        partial
    };
    #[cfg(feature = "multicore")]
    let sums: Vec<_> = if scalars.len() >= 16 && maybe_rayon::current_num_threads() > 1 {
        (0..windows).into_par_iter().map(window_sum).collect()
    } else {
        (0..windows).map(window_sum).collect()
    };
    #[cfg(not(feature = "multicore"))]
    let sums: Vec<_> = (0..windows).map(window_sum).collect();

    sums.into_iter()
        .rev()
        .fold(pallas::Point::identity(), |mut sum, window| {
            for _ in 0..bits {
                sum = sum.double();
            }
            sum + window
        })
}
