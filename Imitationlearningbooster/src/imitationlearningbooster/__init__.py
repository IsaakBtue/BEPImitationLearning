"""Imitation Learning Booster — AMP-augmented goalkeeper training package.

Auto-registers tasks when mjlab discovers this package via entry points.
"""

# Auto-register all tasks when this package is loaded by mjlab
from imitationlearningbooster.tasks import register_all
register_all()
