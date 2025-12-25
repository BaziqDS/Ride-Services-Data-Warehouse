"""
Pages module initialization
Import all page modules here for easier access
"""

from . import data_insertion
from . import analytics_dashboard
from . import ml_predictions
from . import pipeline_control

__all__ = [
    'data_insertion',
    'analytics_dashboard', 
    'ml_predictions',
    'pipeline_control'
]