from my_mjlab_project_lefthand.tasks import register_all
from mjlab.tasks.registry import list_tasks

if "goalkeeper_lefthand" not in list_tasks():
    register_all()


def main() -> None:
    print("my-mjlab-project-lefthand — use 'uv run mjlab train goalkeeper_lefthand' to train.")
