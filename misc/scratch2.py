import os

cmd = "python -m utils.pileup_v{ver} -i /extdata4/baeklab/Hyeonseo/m6A/inference/inference/AIRNA-DW-v2-20240827-175235-22-373000-token_normalise_dwell_all_npz-ON0090_allmotif -o /extdata4/baeklab/Hyeonseo/m6A/inference/inference/AIRNA-DW-v2-20240827-175235-22-373000-token_normalise_dwell_all_npz-ON0090_allmotif_pileup_test"
for ver in [5]+list(range(6, 17)):
    print(f"Running pileup_v{ver}")
    os.system(cmd.format(ver=ver))
