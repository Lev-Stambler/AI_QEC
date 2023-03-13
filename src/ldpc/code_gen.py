import os 
import numpy as np
import subprocess

def gen_code(n, m, dv, seed, construction='peg'):
    filename = '/tmp/code.pchk'
    curr_dir_path = os.path.dirname(os.path.realpath(__file__))
    print(curr_dir_path)
    s = f'cd {curr_dir_path}/../../lib/ProtographLDPC/ && python3 LDPC-library/make-pchk.py --output-pchk-file {filename} --code-type regular --construction {construction} --n-checks {m} --n-bits {n} --checks-per-col {dv} --seed {seed}'
    subprocess.getoutput(s)
    # subprocess.getoutput('cp  .')
    subprocess.getoutput(
        f'(cd {curr_dir_path}/../../lib/ProtographLDPC/ && pchk-to-alist {filename} {filename}.alist)')

    code_file = open(f"{filename}.alist", 'r')
    code_a_list = code_file.read()
    code_file.close()

    lines = code_a_list.split('\n')
    m, n = int(lines[0].split(' ')[0]), int(lines[0].split(' ')[1])
    bit_lines = [[int(i) - 1 for i in s.split(' ') if i != '']
                 for s in lines[4 + m:4+m+n]]
    checks_lines = [[int(i) - 1 for i in s.split(' ') if i != '' and i != '0']
                    for s in lines[4:4+m]]
    checks_lines, bit_lines

    H = np.zeros((m, n), dtype=np.uint8)
    for check in range(len(checks_lines)):
        for bit in checks_lines[check]:
            H[check, bit] = 1

    return H

if __name__ == '__main__':
    H = gen_code(30, 20, 2, 4)
    print(H)