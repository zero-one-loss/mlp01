import time
import os
import pickle
import numpy as np
import sys

def save_checkpoint(obj, save_path, file_name, et, vc):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    suffix = '%s'*6 % et[:6]
    new_name, extend = os.path.splitext(file_name)
    new_name = "%s_%s%s" % (new_name, suffix, extend)
    full_name = os.path.join(os.getcwd(), os.path.join(save_path, file_name))
    with open(full_name, 'wb') as f:
        pickle.dump(obj, f)
    print('Save %s successfully, verification code: %s' % (full_name, vc))


def print_title(vc_len=4):
    vc_table = [chr(i) for i in range(97, 123)]
    vc = ''.join(np.random.choice(vc_table, vc_len))
    print(' ')
    et = time.localtime()
    print('Experiment time: ', time.strftime("%Y-%m-%d %H:%M:%S", et))
    print('Verification code: ', vc)
    print('Args:')
    print(sys.argv)

    return et, vc