"""Small-data run controls; architecture and pretrained loading stay upstream."""
import json
from pathlib import Path
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer

class TumSegPretrainedTrainer(PretrainedTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        import torch
        super().__init__(plans, configuration, fold, dataset_json,
                         device if device is not None else torch.device('cuda'))
        settings = plans['tumseg_experiment']
        self.num_epochs = settings['epochs']
        self.num_iterations_per_epoch = settings['iterations_per_epoch']
        self.num_val_iterations_per_epoch = settings['monitoring_iterations']
        self.warmup_duration_whole_net = settings['warmup_epochs']
        self.initial_lr = 1e-3
        self.save_every = 5

    def do_split(self):
        training, monitoring = super().do_split()
        expected = sorted(self.plans_manager.plans['tumseg_experiment']['training_cases'])
        assert sorted(training) == expected
        assert sorted(monitoring) == expected
        self.print_to_log_file('MONITORING USES TRAINING CASES; NOT INDEPENDENT VALIDATION. Test case is isolated.')
        return training, monitoring

    def save_checkpoint(self, filename):
        if Path(filename).name == 'checkpoint_best.pth':
            # No model selection on resubstitution monitoring or the held-out case.
            return
        return super().save_checkpoint(filename)

    def on_epoch_end(self):
        super().on_epoch_end()
        import torch
        record = dict(completed_epochs=self.current_epoch, total_epochs=self.num_epochs,
                      optimizer_steps=self.current_epoch*self.num_iterations_per_epoch,
                      gpu_peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                      monitor_scope='training cases only')
        path = Path(self.output_folder)/'progress.json'
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
