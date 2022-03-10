import numpy as np

from composition.utils.mjk_utils import xml_path_completion
from composition.arenas.compositional_arena import CompositionalArena


class TrashcanArena(CompositionalArena):
    """
    Workspace that contains two bins placed side by side.

    Args:
        trashcan_pos (3-tuple): (x,y,z) position to place the trashcan
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
    """

    def __init__(
        self, bin1_pos=None, bin2_pos=None
    ):
        super().__init__(xml_path_completion("arenas/trashcan_arena.xml"), bin1_pos=bin1_pos,
                         bin2_pos=bin2_pos)

        self.trashcan_pos = np.array(list(map(float, self.worldbody.find(
            ".//body[@name='trashcan']").items()[1][1].split())))
        self.trashcan_floor_size = np.array(list(map(float, self.worldbody.find(
            ".//geom[@name='trashcan5_collision']").items()[1][1].split())))
        self.trashcan_height = np.array(list(map(float, self.worldbody.find(
            ".//geom[@name='trashcan1_collision']").items()[1][1].split())))[2]