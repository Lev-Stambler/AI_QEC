use core::time;

use ndarray::prelude::*;
use ndarray::{Array, Ix1};

type ErrorPattern = Array<u8, Ix2>;
type Syndrome = Array<u8, Ix2>;
type ParityCheck = Array<u8, Ix2>;

trait Code {
    fn calculate_syndrome(&self, error_pattern: ErrorPattern) -> Syndrome;
}

impl Code for ParityCheck {
    fn calculate_syndrome(&self, error_pattern: ErrorPattern) -> Syndrome {
        &self.dot(&error_pattern) % 2
    }
}

struct BP {
    check_to_bit: Vec<f64>,
    bit_to_check: Vec<f64>,
    // update_check_to_bit: || TODO: MAKE MODULAR LATER
}

/// Implement BP for Code

impl BP {
    fn new() -> Self {
        todo!()
    }
    fn run_bp(
        &mut self,
        time_steps: u32,
        syndrome: Syndrome,
        pc: &ParityCheck,
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
    fn single_step(&self, pc: &ParityCheck, curr_err_pattern: ErrorPattern) -> ErrorPattern {
        for (check, p) in pc.outer_iter().enumerate() {
            for (bit, on) in p.iter().enumerate() {
                if *on == 1 {
                    self.check_to_bit(pc, check, bit)
                }
            }
            // TODO: THERE HAS TO BE A FASTER WAY
            for (bit, on) in p.iter().enumerate() {
                if *on == 1 {
                    self.bit_to_check(pc, check, bit)
                }
            }
        }
        todo!()
    }
    fn decoded_done(&self, err_pattern: &ErrorPattern) -> bool {
        return true;
    }
    fn check_to_bit(&mut self, pc: &ParityCheck, check: usize, bit: usize) {}
    fn bit_to_check(&mut self, pc: &ParityCheck, check: usize, bit: usize) {}
}

fn main() {
    println!("Hello, world!");
}
