# utils/scheduler_utils.py
def rescale_schedule(original_schedule, total_steps):
    rescaled_schedule, max_original_step = {}, 0
    for key, schedule_points in original_schedule.items():
        if schedule_points: max_original_step = max(max_original_step, max(schedule_points.keys()))
    if max_original_step == 0: return original_schedule
    for key, schedule_points in original_schedule.items():
        if not schedule_points: rescaled_schedule[key] = {}; continue
        rescaled_points = {0: schedule_points.get(0, 0.0)}
        for original_step, value in schedule_points.items():
            if original_step == 0: rescaled_points[0] = value; continue
            new_step = int((original_step/max_original_step)*total_steps)
            rescaled_points[new_step] = value
        rescaled_schedule[key] = dict(sorted(rescaled_points.items()))
    return rescaled_schedule