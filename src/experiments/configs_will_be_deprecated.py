from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class DataConfig:
    _target_: str = "src.data.config.DatasetConfig"
    name: str = "edinburghcstr/ami"
    subset: str = "ihm"
    split_name: str = "test"    
    text_column: str = "text" 
    revision: str = "refs/convert/parquet"
    interim_path: str = "data/interim"





# bu parent run, aslında experiment oluyor. manuscriptte anlam ifade eden en küçük bütün.
@dataclass 
class ParentRunConfig:
    experiment_name: str = "Synthetic Data Experiment"
    parent_run_name: str = "synthetic_v1"

    # bu ileride any number of models için genişletilebilir. 
    child_runs: List[Any] = field(default_factory=lambda: [
        ChildRunConfig(name="model_a"),
        ChildRunConfig(name="model_b"),
    ])

    data: DataConfig = field(default_factory=lambda: DataConfig())


