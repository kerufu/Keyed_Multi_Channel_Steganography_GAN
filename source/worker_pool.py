import setting

import dataset_worker
dw = dataset_worker.dataset_worker()
dw.dataset = dw.dataset.take(setting.batch_size)
dw.dataset = dw.dataset.shuffle(dw.dataset.cardinality()//setting.shuffle_buffer_size_divider, reshuffle_each_iteration=True).batch(setting.batch_size, drop_remainder=True)

import GAN_worker
ganw = GAN_worker.GAN_worker(setting.GAN_key)

import AE_worker
aew = AE_worker.AE_worker(setting.AE_key)

import character_mapper
cm = character_mapper.character_mapper()

import botnet
bw = botnet.botnet_worker()

