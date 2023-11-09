import os
os.chdir("..")

import numpy as np

import setting

import dataset_worker
dw = dataset_worker.dataset_worker()

import GAN_worker
ganw = GAN_worker.GAN_worker(setting.GAN_key)
# ganw.train(50000, dw.dataset)
ganw.evaluate(dw.dataset, coding_mode=0)

# import AE_worker
# aew = AE_worker.AE_worker(setting.AE_key)
# aew.train(50000, dw.dataset)

# import botnet
# botnet.botnet_simulation()
# botnet.generate_twitter_profile_image()
# botnet.decode_twitter_profile_image()

