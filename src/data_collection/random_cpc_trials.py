from hashlib import sha1
import os
from CPC import cpc_code
import json
from CPC.generate_random import random_cpc
from common import CPCTrialParams, calculate_tanner_p_error_depolarizing
from ldpc_classical import code_gen
import random
from ldpc_classical.aff3ct_wrapper import aff3ct_simulate
from scoring.score_dataset import run_decoder_bp_osd
import numpy as np
import numpy.typing as npt

def _add_random_cross_checks(cpc: cpc_code.CPCCode, n_cross_checks):
    i = 0
    check_verts = cpc.get_all_check_vertices()
    sample_list = list(range(len(check_verts)))

    while i < n_cross_checks:
        idxs = random.sample(sample_list, 2)
        cross_edge = cpc_code.CPCEdge(
            check_verts[idxs[0]], check_verts[idxs[1]], bit_check=False)
        if not cpc.has_edge(cross_edge):
            cpc.add_edge(cross_edge)
            i += 1

def save_wer(file: str, param: CPCTrialParams, wer):
    wers = {}
    if os.path.isfile(file):
        f = open(file, "r")
        f_content = f.read()
        wers = json.loads(f_content)
        f.close()

    k = param.get_key()
    wers[k] = {}
    wers[k]["wer"] = wer
    wers[k]["params"] = vars(param)
    f = open(file, "w")
    f.write(json.dumps(wers))
    f.close()

# TODO: memo with the codes?


def random_code_trials(params: list[CPCTrialParams], cpc_code_save_dir: str, save_res_file: str):
    out = []
    for param in params:
        cpc_path = f"{cpc_code_save_dir}/{param.get_key()}.cpc"
        cpc = None
        if not os.path.exists(cpc_path):
            cpc = random_cpc(param.n, param.m_x, param.dv_x, param.m_z,
                             param.dv_z, param.seeds[0], param.seeds[1])
            cpc.save(cpc_path)
        else:
            print("Loading")
            cpc = cpc_code.CPCCode.load(cpc_path)
        _add_random_cross_checks(cpc, param.n_cross_checks)
        H, bit_types = cpc.get_tanner_graph()
        err = calculate_tanner_p_error_depolarizing(
            bit_types, param.p_error[0], param.p_error[1], param.p_error[2])
        wer = aff3ct_simulate.get_wer(H, err)
        out.append(wer)
        save_wer(save_res_file, param, wer)
        # cpc_code.BitType.
