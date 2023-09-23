import dataset_worker
import GAN_worker

dw = dataset_worker.dataset_worker()
ganw = GAN_worker.GAN_worker()
# ganw.train(500, dw.dataset)
ganw.test(dw.dataset)