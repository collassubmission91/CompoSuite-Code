import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string


class CompositionalArena(Arena):
    """
    Workspace that contains two bins placed side by side.

    Args:
        trashcan_pos (3-tuple): (x,y,z) position to place the trashcan
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
    """

    def __init__(
        self, fname, bin1_pos=None, bin2_pos=None
    ):
        super().__init__(fname=fname)

        if bin1_pos is None:
            self.bin1_pos = np.array(list(map(float, self.worldbody.find(
                "./body[@name='bin1']").items()[1][1].split())))
        else:
            self.bin1_pos = np.array(bin1_pos)
            self.worldbody.find(
                "./body[@name='bin1']").set("pos", array_to_string(bin1_pos))

        if bin2_pos is None:
            self.bin2_pos = np.array(list(map(float, self.worldbody.find(
                "./body[@name='bin2']").items()[1][1].split())))
        else:
            self.bin2_pos = np.array(bin2_pos)
            self.worldbody.find(
                "./body[@name='bin2']").set("pos", array_to_string(bin2_pos))

        self.bin1_size = np.array(list(map(float, self.worldbody.find(
            "./body[@name='bin1']/geom[@name='floor1']").items()[2][1].split())))
        self.bin2_size = np.array(list(map(float, self.worldbody.find(
            "./body[@name='bin2']/geom[@name='floor2']").items()[2][1].split())))

        self.bin1_wall_right_size = np.array(list(map(float, self.worldbody.find(
            ".//geom[@name='bin1_right_wall']").items()[1][1].split())))
        # self.bin1_wall_left_size = np.array(list(map(float, self.worldbody.find(
        #     ".//geom[@name='bin1_left_wall']").items()[1][1].split())))
        self.bin1_wall_front_size = np.array(list(map(float, self.worldbody.find(
            ".//geom[@name='bin1_front_wall']").items()[1][1].split())))
        self.bin1_wall_rear_size = np.array(list(map(float, self.worldbody.find(
            ".//geom[@name='bin1_rear_wall']").items()[1][1].split())))

        # TODO: do we need this line?
        self.table_top_abs = np.array(bin1_pos)

        self.configure_location()

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))
