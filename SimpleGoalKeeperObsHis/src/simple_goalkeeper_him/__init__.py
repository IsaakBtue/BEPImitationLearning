"""SimpleGoalKeeperHim — HIM-PPO foot-only goalkeeper, 2-disc AMP, 21-DOF headless T1.

Auto-registers the simple_goalkeeper_him task when mjlab discovers this package.
"""

from simple_goalkeeper_him.tasks import register_all
register_all()
