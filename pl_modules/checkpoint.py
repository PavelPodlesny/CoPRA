from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint


class ModelCheckpointWithResults(ModelCheckpoint):
    """ModelCheckpoint that keeps `<output_dir>/pred/<checkpoint stem>.csv` (the validation predictions of
    the epoch the checkpoint was saved at, e.g. `epoch=12-val_loss=0.845.csv`, `last.csv`) in step with the
    checkpoint files: written when a checkpoint is saved, removed when it is rotated out of the top-k.

    Overrides two private ModelCheckpoint methods (pytorch-lightning 2.x).
    """

    @staticmethod
    def _results_path(trainer, filepath):
        pred_dir = trainer.lightning_module.output_dir
        return None if pred_dir is None else Path(pred_dir) / f'{Path(filepath).stem}.csv'

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        path = self._results_path(trainer, filepath)
        results = trainer.lightning_module.val_results
        if trainer.is_global_zero and path is not None and results is not None:
            results.to_csv(path, index=False)

    def _remove_checkpoint(self, trainer, filepath):
        super()._remove_checkpoint(trainer, filepath)
        path = self._results_path(trainer, filepath)
        if trainer.is_global_zero and path is not None:
            path.unlink(missing_ok=True)
