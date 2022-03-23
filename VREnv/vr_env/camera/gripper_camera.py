import numpy as np
import pybullet as p

from vr_env.camera.camera import Camera


class GripperCamera(Camera):
    def __init__(self, fov, aspect, nearval, farval, width, height, robot_id, cid, name, objects=None):
        self.cid = cid
        self.robot_uid = robot_id
        links = {
            p.getJointInfo(self.robot_uid, i, physicsClientId=self.cid)[12].decode("utf-8"): i
            for i in range(p.getNumJoints(self.robot_uid, physicsClientId=self.cid))
        }
        self.gripper_cam_link = links["gripper_cam"]
        self.fov = fov
        self.aspect = aspect
        self.nearval = nearval
        self.farval = farval
        self.width = width
        self.height = height
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=self.aspect, nearVal=self.nearval, farVal=self.farval
        )
        self.name = name
        self.tcp2cam_T = self.get_tcp2cam_transform(links['tcp'])

    def get_tcp2cam_transform(self, tcp_link_id):
        camera_ls = p.getLinkState(
            bodyUniqueId=self.robot_uid, linkIndex=self.gripper_cam_link, physicsClientId=self.cid
        )
        tcp_pos, tcp_orn = p.getLinkState(
            self.robot_uid, tcp_link_id, physicsClientId=self.cid)[:2]

        camera_pos, camera_orn = camera_ls[:2]

        tcp_to_global = p.invertTransform(tcp_pos, tcp_orn)
        tcp_to_cam = p.multiplyTransforms(*tcp_to_global,
                                          camera_pos, camera_orn)
        return list(tcp_to_cam)

    def render(self):
        camera_ls = p.getLinkState(
            bodyUniqueId=self.robot_uid, linkIndex=self.gripper_cam_link, physicsClientId=self.cid
        )
        camera_pos, camera_orn = camera_ls[:2]
        cam_rot = p.getMatrixFromQuaternion(camera_orn)
        cam_rot = np.array(cam_rot).reshape(3, 3)
        cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
        # camera: eye position, target position, up vector
        self.viewMatrix = p.computeViewMatrix(camera_pos, camera_pos + cam_rot_y, -cam_rot_z)
        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            physicsClientId=self.cid,
        )
        rgb_img, depth_img = self.process_rgbd(image, self.nearval, self.farval)
        return rgb_img, depth_img

    def get_projection_matrix(self):
        return self.projectionMatrix
