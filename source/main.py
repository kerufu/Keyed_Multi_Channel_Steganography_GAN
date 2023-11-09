import os
os.chdir("..")

import numpy as np

import worker_factory

# worker_factory.ganw.train(50000, worker_factory.dw.dataset)
worker_factory.ganw.evaluate(worker_factory.dw.dataset, coding_mode=0)

# worker_factory.aew.train(50000, worker_factory.dw.dataset)

# worker_factory.bw.botnet_simulation()
# worker_factory.bw.generate_twitter_profile_image()
# worker_factory.bw.decode_twitter_profile_image()

