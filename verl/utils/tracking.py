# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""
import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Dict, Any


class Tracking(object):
    supported_backend = ['wandb', 'mlflow', 'console']

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import wandb
            import os
            WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
            if WANDB_API_KEY:
                wandb.login(key=WANDB_API_KEY)
            
            # 配置 wandb
            wandb_config = {
                'project': project_name,
                'name': experiment_name,
                'config': config,
                'settings': wandb.Settings(
                    start_method="thread",
                    _disable_stats=True
                )
            }
            
            # 显式确保 'watch' 和 'log_model' 不会作为顶层参数传递给 wandb.init()
            # 这些功能应该通过环境变量或 wandb.watch() / wandb.log_artifact() 单独处理
            if 'watch' in wandb_config:
                del wandb_config['watch']
            if 'log_model' in wandb_config:
                del wandb_config['log_model']
            
            wandb.init(**wandb_config)
            
            # 如果配置了 watch，提示用户在模型实例化后手动调用 wandb.watch()
            if config and 'trainer' in config and config['trainer'].get('wandb_watch') and hasattr(wandb, 'watch') and callable(wandb.watch):
                print(f"W&B: wandb.watch() is configured via config.trainer.wandb_watch='{config['trainer']['wandb_watch']}'.")
                print(f"W&B: Please ensure wandb.watch(model) is called after model initialization to enable automatic logging of gradients and parameters.")

            self.logger['wandb'] = wandb

        if 'mlflow' in default_backend:
            import mlflow
            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger['mlflow'] = _MlflowLoggingAdapter()

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                # 对于 wandb，确保所有指标都被记录
                if default_backend == 'wandb':
                    # 由于这里没有访问模型的方法，我们移除这部分代码
                    # 这些功能将通过 wandb.watch() 自动完成
                    pass
                
                logger_instance.log(data=data, step=step)


class _MlflowLoggingAdapter:

    def log(self, data, step):
        import mlflow
        mlflow.log_metrics(metrics=data, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep='/')


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {'list_len': len(x)} | {f'{i}': _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient='records')[0]
    assert isinstance(ans, dict)
    return ans
