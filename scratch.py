def get_min_ideal_displacement_dict(cb_per_bb, spacer_size, cb_size):
    min_ideal_displacement_dict = {}
    big_step_size = cb_size + spacer_size
    small_step_size = spacer_size

    for from_idx in range(cb_per_bb+1):
        for to_idx in range(cb_per_bb+1):
            if from_idx < to_idx:
                small_steps = 0
                big_steps = to_idx - from_idx
                displacement = big_step_size * big_steps
            else:
                small_steps = 1
                big_steps = cb_per_bb - from_idx + to_idx
                displacement = big_step_size * big_steps + small_step_size
            min_ideal_displacement_dict[(from_idx, to_idx)] = (displacement,small_steps,big_steps)

    return min_ideal_displacement_dict

print(get_min_ideal_displacement_dict(3, 6, 21))
