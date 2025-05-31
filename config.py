class Config:
    def __init__(self, epochs, loss, lr, batch_size):
        self.epochs = epochs
        self.loss = loss,
        self.lr = lr
        self.batch_size = batch_size
        self.loss_fn = "contrastive"
