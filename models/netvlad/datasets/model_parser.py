"""Helper class to parse ground truth 3D models.
"""
import os
import numpy as np
from .internal import read_model
from ..pose_prediction.matrix_utils import assemble_matrix
from ..pose_prediction.matrix_utils import quaternion_matrix
from typing import Dict, Tuple, List

QuaternionAndPose = Tuple[np.ndarray, np.ndarray]
PoseDictionary = Dict[str, QuaternionAndPose]
IdToSet = Dict[int, List[int]]


class ModelParser(object):

    def __init__(self, filename_to_pose: PoseDictionary,
                       camera_to_points: IdToSet = None,
                       point_to_cameras: IdToSet = None):

        """Initialize model parser.

        Args:
            filename_to_pose: A dictionary mapping a filename to a camera
                orientation stored as a quaternion, and a camera pose matrix.
            camera_to_points: (Optional) A dictionary mapping a camera id to the
                set of visible 3D points IDs.
            point_to_cameras: (Optional) A dictionary mapping a 3D point ID to
                the list of camera IDs it is visible in.
        """
        self._filename_to_pose = filename_to_pose
        self._camera_to_points = camera_to_points
        self._point_to_cameras = point_to_cameras
        self._filenames = list(filename_to_pose.keys())

    @classmethod
    def from_nvm(cls, nvm_path: str):
        """ Parse ground truth poses from an .nvm file.

        Args:
            nvm_path: The path to the .nvm file.
        Returns:
            model_parser: An instance of ModelParser.
        """
        # Parse .nvm file
        nvm_poses = [line.rstrip('\n') for line in open(nvm_path)]
        # Load poses and reference filenames from the .NVM file
        fname_to_pose = dict()
        for i in range(int(nvm_poses[2])):
            # Parse quaternions and camera positions
            l = nvm_poses[i+3].split(' ')
            f = '/'.join(
                l[0].rstrip('\n').split('/')[1:4]).replace('png', 'jpg')
            q = [float(x) for x in l[2:6]]
            c = [float(x) for x in l[6:9]]
            R = quaternion_matrix(q)[:3,:3]
            M = assemble_matrix(R, c)
            M[:3,3] = np.matmul(-M[:3,:3],M[:3,3])
            fname_to_pose[f] = [q, M]

        model_parser = cls(fname_to_pose)
        return model_parser

    @classmethod
    def from_binary(cls, bin_path: str):
        """ Parse ground truth poses from a .bin file.

        Args:
            bin_path: The path to the .bin file.
        Returns:
            model_parser: An instance of ModelParser.
        """
        _, images, _ = read_model.read_model(
            os.path.splitext(bin_path)[0], '.bin')
        fname_to_pose = dict()
        for _, x in images.items():
            R = quaternion_matrix(x.qvec)[:3,:3]
            M = assemble_matrix(R, x.tvec)
            fname_to_pose[x.name] = [x.qvec, M]

        model_parser = cls(fname_to_pose)
        return model_parser

    @property
    def filename_to_pose(self):
        return self._filename_to_pose

    @property
    def camera_to_points(self):
        return self._camera_to_points

    @property
    def point_to_cameras(self):
        return self._point_to_cameras

    @property
    def filenames(self):
        return self._filenames
