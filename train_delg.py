#!/DataA/fyp22_jiadong_guo/ .conda/envs/test1/bin/python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# written by yangmin09

#import os
import core.config as config
import core.distributed as dist
import core.trainer as trainer
from core.config import cfg


def main():
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    config.load_cfg_fom_args("Train a delg model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)
    # trainer.train_model()


if __name__ == "__main__":
    main()
