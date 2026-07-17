"""日志工具函数"""
import os
import sys
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logger(name: str = 'defect_detection', 
                 log_dir: str = 'logs',
                 level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'{name}_{timestamp}.log')
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    def __init__(self, experiment_name: str, output_dir: str = 'experiments'):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(experiment_name, str(self.output_dir / 'logs'))
        self.metrics_history = []
        self.config = None
        self.start_time = time.time()
    
    def log_config(self, config: Dict[str, Any]):
        self.config = config
        config_path = self.output_dir / 'config.yaml'
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        self.logger.info(f'Config saved to {config_path}')
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        record = {'epoch': epoch, 'timestamp': time.time(), **metrics}
        self.metrics_history.append(record)
        self.logger.info(f'Epoch {epoch}: ' + 
                        ', '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                                  for k, v in metrics.items()))
    
    def save_results(self):
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'duration': time.time() - self.start_time,
                'config': self.config,
                'metrics': self.metrics_history,
            }, f, indent=2, ensure_ascii=False)
        self.logger.info(f'Results saved to {results_path}')
    
    def log_message(self, message: str, level: str = 'info'):
        getattr(self.logger, level)(message)


class Timer:
    def __init__(self, name: str = ''):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            print(f'{self.name}: {self.elapsed:.3f}s')
