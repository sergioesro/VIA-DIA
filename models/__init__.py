"""
Model registry for DIA project.

Available architectures:
- BaselineMultimodalModel: Simple CNN+MLP fusion (fast, no attention)
- ImageDifferenceCNNModel: Image-only CNN feature-difference classifier
"""

from .baseline_model import BaselineMultimodalModel, create_baseline_model
from .image_difference_model import ImageDifferenceCNNModel, create_image_difference_model

__all__ = [
    'BaselineMultimodalModel',
    'create_baseline_model',
    'ImageDifferenceCNNModel',
    'create_image_difference_model',
]


def create_model(arch_name, **kwargs):
    """
    Factory function to create models by name.
    
    Parameters
    ----------
    arch_name : str
        One of: 'BaselineMultimodalModel', 'ImageDifferenceCNNModel'
    **kwargs : dict
        Model-specific parameters
        
    Returns
    -------
    model : nn.Module
    """
    if arch_name == 'BaselineMultimodalModel':
        return create_baseline_model(**kwargs)
    elif arch_name == 'ImageDifferenceCNNModel':
        return create_image_difference_model(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
