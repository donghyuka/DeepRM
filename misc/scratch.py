import re
import numpy as np
import time

INCREMENTS_CIGAR = {
    'M': [1, 1],
    'I': [0, 1],
    'S': [0, 1],
    'D': [1, 0],
    'N': [1, 0],
}

def ref_pos_to_query_pos(ref_pos_request_arr, cigar, start_pos):
    x = [[start_pos - 1, -1]]
    for length, op in re.findall(r'(\d+)([MIDSN])', cigar):
        x += [INCREMENTS_CIGAR[op]] * int(length)

    x = np.array(x)
    x = np.cumsum(x, axis=0)[np.sum(x, axis=1) == 2]
    query_pos_list = []

    ref_pos_request_idx = 0
    ref_pos_request_idx_max = len(ref_pos_request_arr)
    exit_flag = False
    for row in x:
        while True:
            ref_pos_request = ref_pos_request_arr[ref_pos_request_idx]
            ref_pos, query_pos = row
            if ref_pos == ref_pos_request:
                ## Found the requested ref_pos. Register the query_pos. Go fetch next request AND next ref_pos.
                query_pos_list.append(query_pos)
                if ref_pos_request_idx < ref_pos_request_idx_max - 1:
                    ref_pos_request_idx += 1
                    break
                else:
                    exit_flag = True
                    break
            elif ref_pos > ref_pos_request:
                ## Skipped the requested ref_pos. Go fetch next request.
                query_pos_list.append(None)
                if ref_pos_request_idx < ref_pos_request_idx_max - 1:
                    ref_pos_request_idx += 1
                else:
                    exit_flag = True
                    break
            else:
                ## ref_pos < ref_pos_request
                ## Not yet reached the requested ref_pos. Go fetch next ref_pos.
                break

        if exit_flag:
            break

    return query_pos_list

x = ref_pos_to_query_pos(np.arange(1,6000,1),
                         "37M3D118M1D2M1I438M2I25M4I31M1D181M1D57M1I95M1I2M1D123M1I176M1D31M2D117M2D122M2D34M27S",
                         4604)

print(x)
print(len(x))