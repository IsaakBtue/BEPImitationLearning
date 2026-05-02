from my_mjlab_project.tasks import register_all
from mjlab.tasks.registry import list_tasks

# Only register if not already registered (allows reimporting)
if "goalkeeper" not in list_tasks():
    register_all()


def main() -> None:
    print("my-mjlab-project — use 'uv run mjlab train goalkeeper' to train.")
