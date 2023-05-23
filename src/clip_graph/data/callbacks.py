import pytorch_lightning as pl

class RebuildContrastiveDataset(pl.Callback):
    def on_train_epoch_end(self, trainer, model):
        trainer.datamodule.train_dataset.next_epoch()
