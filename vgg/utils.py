import math
from datetime import datetime

def imread(path):
    import scipy.misc.imread
    return scipy.misc.imread(path).astype(np.float)

def time_tensorflow_run(session, target, num_batches, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in xrange(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                             (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
                 (datetime.now(), info_string, num_batches, mn, sd))
