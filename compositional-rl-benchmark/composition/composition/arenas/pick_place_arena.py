import numpy as np

from composition.arenas.compositional_arena import CompositionalArena
from composition.utils.mjk_utils import xml_path_completion

from robosuite.utils.mjcf_utils import array_to_string

class PickPlaceArena(CompositionalArena):
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
        super().__init__(xml_path_completion("arenas/pick_place_arena.xml"), bin1_pos=bin1_pos,
                         bin2_pos=bin2_pos)

        
