import os 
import robosuite
from composition.tasks.pick_place_subtask import PickPlaceSubtask
from composition.tasks.push_subtask import PushSubtask
from composition.tasks.shelf_subtask import ShelfSubtask
from composition.tasks.trashcan_subtask import TrashcanSubtask

robosuite.environments.base.register_env(PickPlaceSubtask)
robosuite.environments.base.register_env(PushSubtask)
robosuite.environments.base.register_env(ShelfSubtask)
robosuite.environments.base.register_env(TrashcanSubtask)

from composition.env.main import make
assets_root = os.path.join(os.path.dirname(__file__), "assets")
