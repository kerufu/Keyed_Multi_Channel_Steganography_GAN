import os
os.chdir("..")

import dataset_worker
import GAN_worker
import botnet

# dw = dataset_worker.dataset_worker()
# ganw = GAN_worker.GAN_worker()

# ganw.train(50000, dw.dataset)
# ganw.evaluate(dw.dataset, coding_mode=0)

# botnet.botnet_simulation()

# botnet.generate_twitter_profile_image()
botnet.decode_twitter_profile_image()

