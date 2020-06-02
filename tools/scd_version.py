import numpy as np
import pickle
import sys
sys.path.append('..')
# from tools import args
#  scd version control

def scd_version_control(version, args):
    """

    :param version: scd algorithm version
    :return: scd class
    """

    if version == 'v1':
        from core.scd_v9_gpu import SCD
        # basic scd 01 loss linear classifier

    elif version == 'v2':
        from core.scd_v15_gpu import SCD
        # scd mlp classifier

    elif version == 'v3':
        from core.debug_v1 import SCD
        # scd adversarial training, randomly 10% data as clean
        # and the same adversarial

    elif version == 'v4':
        from core.debug_v2 import SCD
        # scd 01 adversarial training, top 10% data which closes
        # to decision boundary as clean
        # and the same adversarial

    elif version == 'v5':
        from core.debug_v3 import SCD
        # scd mlp 01 adversarial training, randomly 10% data as clean
        # and the same adversarial

    elif version == 'v6':
        from core.debug_v4 import SCD
        # scd mlp 01 adversarial training, top 10% data which closes
        # to decision boundary as clean
        # and the same adversarial

    elif version == 'v7':
        from core.debug_v5 import SCD
        # scd adversarial training, randomly 10% data as clean
        # and the same adversarial (Majority Vote)

    elif version == 'v8':
        from core.debug_v6 import SCD
        # scd 01 adversarial training, top 10% data which closes
        # to decision boundary as clean
        # and the same adversarial (Majority Vote)

    elif version == 'v9':
        from core.debug_v7 import SCD
        # scd mlp 01 adversarial training, randomly 10% data as clean
        # and the same adversarial (Majority Vote)

    elif version == 'v10':
        from core.debug_v8 import SCD
        # scd mlp 01 adversarial training, top 10% data which closes
        # to decision boundary as clean
        # and the same adversarial (Majority Vote)

    return scd




if __name__ == '__main__':

    from tools import args

    scd_params = {
        'nrows': args.nrows,
        'nfeatures' : args.nfeatures,
        'w_inc': args.w_inc,
        'tol': 0.00000,
        'local_iter': args.iters,
        'num_iters': args.num_iters,
        'interval': args.interval,
        'round': args.round,
        'w_inc1': args.w_inc1,
        'updated_features': args.updated_features,
        'n_jobs': args.n_jobs,
        'num_gpus': args.num_gpus,
        'adv_train': True if args.adv_train else False,
        'eps': args.eps,
        'w_inc2': args.w_inc2,
        'hidden_nodes': args.hidden_nodes,
        'evaluation': False if args.no_eval else True,
        'verbose': True if args.verbose else False,
    }



    scd = scd_version_control('v1', **scd_params)