from typing import List, Union
import numpy as np


def cast_inputs(x: Union[float, List[float], np.ndarray]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif not isinstance(x, list):
        x = [x]
    return np.array(x)
