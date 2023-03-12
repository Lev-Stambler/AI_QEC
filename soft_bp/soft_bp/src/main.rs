use core::time;

use ndarray::prelude::*;
use ndarray::{Array, Ix1};

type ErrorPattern = Array<u8, Ix2>;
type Syndrome = Array<u8, Ix2>;

struct Code {
    pc: Array<u8, Ix2>,
    adj_bit: Vec<Vec<usize>>,
    adj_check: Vec<Vec<usize>>,
    n: usize,
    m: usize,
}

impl Code {
    fn new(pc: Array<u8, Ix2>) -> Code {
        let adj_bit = vec![Vec::new(); pc.shape()[1]];
        let adj_check = vec![Vec::new(); pc.shape()[0]];

        for check in 0..pc.shape()[0] {
            for bit in 0..pc.shape()[1] {
                if let Some(&a) = pc.get((check, bit)) {
                    if a == 1 {
                        adj_bit[bit].push(check);
                        adj_check[check].push(bit);
                    }
                }
            }
        }

        Code {
            pc,
            adj_bit,
            adj_check,
            n: adj_bit.len(),
            m: adj_check.len(),
        }
    }

    fn calculate_syndrome(&self, error_pattern: ErrorPattern) -> Syndrome {
        &self.pc.dot(&error_pattern) % 2
    }
}

struct BP {
    check_total: Vec<f64>,
    bit_total: Vec<f64>,
    // check_to_bit: Vec<Vec<f64>>,
    // bit_to_check: Vec<Vec<f64>>,
    iteration: usize,
    // update_check_to_bit: || TODO: MAKE MODULAR LATER
}

/// Implement BP for Code

impl BP {
    fn new(code: &Code, llr: f64) -> Self {
        BP {
            iteration: 0,
            check_total: vec![0.; code.m],
            bit_total: vec![llr; code.n],
            // bit_to_check: vec![llr; code.n],
            // check_to_bit:
        }
    }
    fn run_bp(
        &mut self,
        time_steps: u32,
        syndrome: Syndrome,
        pc: &Code,
        qubit_error_llr: Vec<f32>,
    ) -> Option<ErrorPattern> {
        let mut err_pattern = Array::<u8, Ix2>::zeros((qubit_error_llr.len(), 1));
        for _ in 0..time_steps {
            err_pattern = self.single_step(pc, err_pattern);
            if self.decoded_done(&err_pattern) {
                // TODO:
                todo!()
            }
        }
        None
    }
    // TODO: this is not particularly efficient but should be fine for out use case?
    fn single_step(&self, pc: &Code, curr_err_pattern: ErrorPattern) -> ErrorPattern {
        // for (check, p) in pc.outer_iter().enumerate() {
        //     for (bit, on) in p.iter().enumerate() {
        //         if *on == 1 {
        //             self.check_to_bit(pc, check, bit)
        //         }
        //     }
        //     // TODO: THERE HAS TO BE A FASTER WAY
        //     for (bit, on) in p.iter().enumerate() {
        //         if *on == 1 {
        //             self.bit_to_check(pc, check, bit)
        //         }
        //     }
        // }
        todo!()
    }
    fn decoded_done(&self, err_pattern: &ErrorPattern) -> bool {
        return true;
    }
    fn check_to_bit(&self, pc: &Code, check: usize, bit: usize) -> f64 {
        self.total
        todo!()
    }
    fn bit_to_check(&mut self, pc: &Code, check: usize, bit: usize) {}
}

fn main() {
    println!("Hello, world!");
}
