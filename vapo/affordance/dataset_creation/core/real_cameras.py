import numpy as np
from robot_io.cams.camera import Camera


class CamProjections(Camera):
    def __init__(
        self,
        intrinsics,
        extrinsic_matrix,
        resolution,
        crop_coords,
        resize_resolution,
        name,
        proj_matrix=None,
    ):
        super().__init__(
            resolution=resolution,
            crop_coords=crop_coords,
            resize_resolution=resize_resolution,
            name=name,
        )
        self.instrinsics = intrinsics
        self.extrinsic_matrix = extrinsic_matrix
        self.intrinsic_matrix = np.array(
            [
                [intrinsics["fx"], 0, intrinsics["cx"]],
                [0, intrinsics["fy"], intrinsics["cy"]],
                [0, 0, 1],
            ]
        )
        if proj_matrix is not None:
            self.projection_matrix = proj_matrix
        else:
            T_cam_world = np.linalg.inv(self.extrinsic_matrix)
            self.projection_matrix = self.intrinsic_matrix @ T_cam_world[:-1, :]
        self.crop_coords = intrinsics["crop_coords"]
        self.resize_resolution = intrinsics["resize_resolution"]
        self.dist_coeffs = intrinsics["dist_coeffs"]
        self.width = self.instrinsics["width"]
        self.height = self.instrinsics["height"]

    def get_intrinsics(self):
        return self.instrinsics

    def get_projection_matrix(self):
        # intr = self.get_intrinsics()
        # cam_mat = np.array([[intr['fx'], 0, intr['cx'], 0],
        #                     [0, intr['fy'], intr['cy'], 0],
        #                     [0, 0, 1, 0]])
        cam_mat = self.projection_matrix
        return cam_mat
