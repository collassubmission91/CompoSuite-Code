import numpy as np

from composition.utils.mjk_utils import xml_path_completion
from composition.arenas.compositional_arena import CompositionalArena

from robosuite.utils.mjcf_utils import array_to_string


class ShelfArena(CompositionalArena):
    """
    Workspace that contains two bins placed side by side.

    Args:
        shelf_pos (3-tuple): (x,y,z) position to place the shelf
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
    """

    def __init__(
        self, bin1_pos=None, bin2_pos=None
    ):
        super().__init__(xml_path_completion("arenas/shelf_arena.xml"), bin1_pos=bin1_pos,
                         bin2_pos=bin2_pos)

        self.shelf_top_size = np.array(list(map(float, self.worldbody.find(
            ".//geom[@name='shelf_top_collision']").items()[1][1].split())))
        self.shelf_side_size = np.array(list(map(float, self.worldbody.find(
            ".//geom[@name='shelf_collision']").items()[1][1].split())))

        self.shelf_bottom_pos = np.array(list(map(float, self.worldbody.find(
            ".//geom[@name='shelf_bottom_collision']").items()[0][1].split())))
        self.shelf_top_pos = np.array(list(map(float, self.worldbody.find(
            ".//geom[@name='shelf_top_collision']").items()[0][1].split())))
        self.shelf_side_pos = np.array(list(map(float, self.worldbody.find(
            ".//geom[@name='shelf_collision']").items()[0][1].split())))

        self.shelf_pos = bin2_pos + np.array([0, self.shelf_top_size[1] + 0.05, 0])
        self.worldbody.find(
            "./body[@name='shelf']").set("pos", array_to_string(self.shelf_pos))


        self.shelf_x_size = self.shelf_top_size[0]
        self.shelf_y_size = self.shelf_top_size[1]
        self.shelf_thickness = self.shelf_top_size[2]
        self.shelf_z_offset = self.shelf_bottom_pos[2]
        self.shelf_z_height = self.shelf_side_size[2]
