import os
os.chdir("..")

import worker_pool

# worker_pool.ganw.train(50000)
# worker_pool.ganw.evaluate(coding_mode=0)
# worker_pool.ganw.plot()

worker_pool.bw.botnet_simulation(character_mapping=False)
# worker_pool.bw.update_twitter_profile_image()
# worker_pool.bw.decode_twitter_profile_image()