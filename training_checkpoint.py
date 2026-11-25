"""Checkpoint and failure diagnostics for ordinary classification training."""
import json
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from timm.utils import get_state_dict

import utils


class TrainingCheckpoint:
    """Own training snapshots without coupling the epoch loop to its caller."""

    def __init__(self, output_dir, model, optimizer, scheduler, scaler, model_ema, args):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.model_ema = model_ema
        self.args = args
        self.best_accuracy = -math.inf

    def _snapshot(self, epoch):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'model_ema': get_state_dict(self.model_ema) if self.model_ema is not None else None,
            'epoch': epoch,
            'best_accuracy': self.best_accuracy,
            'args': self.args,
        }

    def _write(self, checkpoint, path):
        # Only the main rank writes. Replace a complete temporary file atomically.
        temporary = path.with_name('.' + path.name + '.tmp')
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    def restore_best(self, checkpoint):
        """Restore the running best, including legacy checkpoints with a run log."""
        if 'best_accuracy' in checkpoint:
            best = float(checkpoint['best_accuracy'])
        else:
            log_path = self.output_dir / 'log.txt'
            if not log_path.is_file():
                raise ValueError('Legacy resume needs log.txt to recover historical best accuracy')
            records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
            accuracies = [float(row['test_acc1']) for row in records
                          if row['epoch'] <= checkpoint['epoch']]
            if not accuracies:
                raise ValueError('No validation history through the requested resume epoch')
            best = max(accuracies)
        if not math.isfinite(best) or not 0 <= best <= 100:
            raise ValueError('Restored best accuracy must be finite and in [0, 100]')
        self.best_accuracy = best

    def save_epoch(self, epoch, accuracy):
        """Save the epoch and replace the best only on a finite improvement."""
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
            raise ValueError('Validation accuracy must be finite and in [0, 100]')
        improved = accuracy > self.best_accuracy
        self.best_accuracy = max(self.best_accuracy, accuracy)
        if not utils.is_main_process():
            return
        checkpoint = self._snapshot(epoch)
        self._write(checkpoint, self.output_dir / f'checkpoint_{epoch}.pth')
        if improved:
            self._write(checkpoint, self.output_dir / 'checkpoint_best.pth')
        # Retain three ordinary epochs; failure snapshots and best are independent.
        old_checkpoint = self.output_dir / f'checkpoint_{epoch - 3}.pth'
        if epoch >= 3 and old_checkpoint.is_file():
            old_checkpoint.unlink()

    def save_error(self, epoch, batch_index, rank_losses):
        """Persist the interrupted state and diagnostics before the caller raises."""
        if not utils.is_main_process():
            return
        checkpoint_path = self.output_dir / f'erro_{epoch}.pth'
        log_path = self.output_dir / f'erro_{epoch}.log'
        failure = {
            'reason': 'Non-finite training loss; interrupted before backward/optimizer update',
            'time_utc': datetime.now(timezone.utc).isoformat(),
            'epoch': epoch,
            'batch_index': batch_index,  # Zero-based, as is epoch.
            'rank_losses': rank_losses,
            'learning_rates': [group['lr'] for group in self.optimizer.param_groups],
            'checkpoint': str(checkpoint_path.resolve()),
        }
        # Write and flush diagnostics first, so a failed checkpoint write leaves evidence.
        with log_path.open('a', encoding='utf-8') as log:
            log.write(json.dumps(failure, indent=2, allow_nan=False) + '\n')
            log.write('Training configuration:\n' + json.dumps(vars(self.args), indent=2) + '\n')
            log.write('Call stack at interruption:\n' + ''.join(traceback.format_stack()))
            log.write('checkpoint_save_started\n')
            log.flush()
            checkpoint = self._snapshot(epoch)
            checkpoint['failure'] = failure
            self._write(checkpoint, checkpoint_path)
            log.write('checkpoint_saved; training will terminate with FloatingPointError\n')
            log.flush()
        print(f'Non-finite loss: saved {checkpoint_path}; diagnostics: {log_path}', flush=True)
