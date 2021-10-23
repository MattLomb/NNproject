from CLIP.CLIP import CLIP

clip = CLIP(False)
epoches = 10
batch_size = 64
clip.train(epoches, batch_size)
