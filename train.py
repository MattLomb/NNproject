from CLIP.CLIP import CLIP

#Training the CLIP network
clip = CLIP(False)
epoches = 10        # more than 30 minutes per epoch
batch_size = 64
clip.train(epoches, batch_size)
