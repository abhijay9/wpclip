

model = dict(
    vision_backbone = "ViT-B/32"
)

train = dict(
    nEpochs = 10,
    train_batch_size = 4,
    # nThreads = 4,
    valFreq = 50, # every n iterations
    val_batch_size = 1,
)

test = dict(
    test_batch_size = 1
)