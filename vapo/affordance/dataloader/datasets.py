import json
import os

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import rotate

import vapo.affordance.utils.flowlib as flowlib
from vapo.affordance.utils.img_utils import create_circle_mask, overlay_flow, overlay_mask, resize_center, tresh_np
from vapo.affordance.utils.utils import get_abs_path, get_transforms


def get_loaders(logger, dataset_cfg, dataloader_cfg):
    train = VREnvData(split="training", log=logger, **dataset_cfg)
    val = VREnvData(split="validation", log=logger, **dataset_cfg)
    logger.info("train_data {}".format(train.__len__()))
    logger.info("val_data {}".format(val.__len__()))

    train_loader = DataLoader(train, shuffle=True, **dataloader_cfg)
    val_loader = DataLoader(val, **dataloader_cfg)
    logger.info("train minibatches {}".format(len(train_loader)))
    logger.info("val minibatches {}".format(len(val_loader)))
    return train_loader, val_loader, val.out_shape, train.n_classes


class VREnvData(Dataset):
    # split = "train" or "validation"
    def __init__(
        self,
        data_dir,
        img_resize,
        transforms_cfg,
        n_train_ep=-1,
        split="train",
        cam="static",
        log=None,
        radius=None,
    ):
        self.cam = cam
        self.split = split
        self.log = log
        self.root_dir = get_abs_path(data_dir)
        _data_info = self.read_json(os.path.join(self.root_dir, "episodes_split.json"))
        self.data = self._get_split_data(_data_info, split, cam, n_train_ep)
        self.add_rotation = False
        self.img_resize = img_resize
        self.transforms = get_transforms(transforms_cfg[split], img_resize[cam])
        self.mask_transforms = get_transforms(transforms_cfg["masks"], img_resize[cam])
        self.pixel_indices = np.indices((img_resize[cam], img_resize[cam]), dtype=np.float32).transpose(1, 2, 0)
        self.radius = radius
        if cam == "all":
            for cam_type in self.radius.keys():
                self.radius[cam_type] = radius[cam_type] * img_resize[cam] // img_resize[cam_type]
        self.out_shape = self.get_channels(img_resize[cam])
        # Excludes background
        self.n_classes = _data_info["n_classes"] if cam == "static" else 1

    def get_channels(self, in_size):
        test_tensor = torch.zeros((3, in_size, in_size))
        test_tensor = self.transforms(test_tensor)
        return test_tensor.shape  # C, H, W

    def _get_split_data(self, data, split, cam, n_train_ep):
        split_data = []
        split_episodes = list(data[split].keys())

        # Select amount of data to train on
        if n_train_ep > 0 and split == "train":
            assert len(split_episodes) >= n_train_ep, "n_train_ep must <= %d" % len(split_episodes)
            split_episodes = np.random.choice(split_episodes, n_train_ep, replace=False)

        print("%s episodes: %s" % (split, str(split_episodes)))
        for ep in split_episodes:
            data[split][ep].sort()
            for file in data[split][ep]:
                if cam in file or cam == "all":
                    split_data.append("%s/%s" % (ep, file))
        print("%s images: %d" % (split, len(split_data)))
        return split_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # directions: optical flow image in middlebury color

        head, filename = os.path.split(self.data[idx].replace("\\", "/"))
        episode, cam_folder = os.path.split(head)
        data = np.load(self.root_dir + "/%s/data/%s/%s.npz" % (episode, cam_folder, filename))

        # Images are stored in BGR
        frame = data["frame"]
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # C, W, H
        frame = self.transforms(frame)

        # Segmentation mask (H, W)
        # mask = data["mask"]

        # centers, center_dirs = self.get_directions(mask)
        center_dirs, mask = self.label_directions(data["centers"], data["mask"], cam_folder[:-4])
        # center_dirs = torch.tensor(data["directions"]).permute(2, 0, 1)

        # Resize from torchvision requires mask to have channel dim
        # mask = mask.any(axis=0).astype("uint8") * 255  # Binary
        mask = np.vstack([np.zeros((1, *mask.shape[1:])), mask])
        mask = mask.argmax(axis=0).astype("int32")
        mask = np.expand_dims(mask, 0)
        # mask = self.mask_transforms(torch.from_numpy(mask)).long()

        # Rotation transform
        if self.add_rotation:
            res = self.rotate_img(data, frame, mask, center_dirs)
            if res is not None:
                frame, mask, center_dirs = res

        # CE Loss requires mask in form (B, H, W), so remove channel dim
        mask = mask.squeeze()  # H, W

        labels = {
            "affordance": torch.tensor(mask).long(),
            # "centers": centers,
            "center_dirs": center_dirs,
        }
        return frame, labels

    def rotate_img(self, data, frame, mask, directions):
        if self.cam == "gripper" and self.split == "train":
            rotate_frame = np.random.rand() < 0.30  # 30% chance of rotation
            if rotate_frame:
                rand_angle = np.random.randint(-90, 90)
                W, H = data["mask"].shape
                directions = np.stack([np.ones((H, W)), np.zeros((H, W))], axis=-1).astype(np.float32)

                center = data["centers"][0]
                # center_dirs = np.zeros((H, W))
                # center_dirs = torch.tensor(center_dirs).unsqueeze(0)
                # center_dirs[0, center[0], center[1]] = 1
                center_dirs = torch.tensor(np.indices((H, W)))
                center_dirs = rotate(center_dirs, rand_angle)
                indices = center_dirs.numpy()
                indices = np.transpose(indices, (1, 2, 0))
                center = np.argwhere((indices == center).all(-1))
                # center = np.where(center_dirs[0] == 1)
                # center = np.array([c for c in center]).transpose()
                if len(center) <= 0:  # Ony rotate if center is inside img
                    return None

                # gripper always have just one center
                mask = rotate(mask, rand_angle)
                np_mask = mask.squeeze().numpy() * 255
                directions, mask = self.label_directions(center[0].transpose(), np_mask, directions, "gripper")
                # Rotate
                frame = rotate(frame, rand_angle)
                directions = torch.tensor(directions).permute(2, 0, 1)
                # center_dirs = rotate(center_dirs, rand_angle)
        return (frame, mask, directions)

    def label_directions(self, centers, object_mask, curr_cam):
        """
        centers: np.array(shape=(1, 2), dtype='int64') pixel indices in orig. shape
        object_mask: np.array(shape=(n_classes, 2, 2), dtype='uint8') 0-255
        """
        n_classes = object_mask.shape[0]
        old_shape = object_mask.shape[1:]
        new_shape = self.pixel_indices.shape
        direction_labels = np.stack([np.ones(new_shape[:-1]), np.zeros(new_shape[:-1])], axis=-1).astype(np.float32)
        obj_mask = np.zeros((n_classes, *new_shape[:-1]))
        indices = self.pixel_indices
        r = self.radius[curr_cam]

        # return blank masks and directions
        if len(centers) == 0:
            return direction_labels, obj_mask

        for center_info in centers:
            center = center_info[1:]
            label = center_info[0]
            c = resize_center(center, old_shape, new_shape[:2])
            object_center_directions = (c - indices).astype(np.float32)
            object_center_directions = object_center_directions / np.maximum(
                np.linalg.norm(object_center_directions, axis=2, keepdims=True), 1e-10
            )

            # Add it to the labels
            new_mask = create_circle_mask(obj_mask[label], c[::-1], r=r)
            new_mask = tresh_np(new_mask, 100).squeeze()
            direction_labels[new_mask == 1] = object_center_directions[new_mask == 1]
            obj_mask[label] = overlay_mask(new_mask, obj_mask[label], (255, 255, 255))

        direction_labels = torch.tensor(direction_labels).permute(2, 0, 1)
        return direction_labels.float(), obj_mask * 255

    def read_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data


def test_dir_labels(hv, frame, aff_mask, center_dir):
    bool_mask = (aff_mask == 1).int().cuda()
    center_dir = center_dir.cuda()  # 1, 2, H, W
    initial_masks, num_objects, object_centers_padded = hv(bool_mask, center_dir.contiguous())

    initial_masks = initial_masks.cpu()
    object_centers_padded = object_centers_padded[0].cpu().permute((1, 0))
    for c in object_centers_padded:
        c = c.detach().cpu().numpy()
        u, v = int(c[1]), int(c[0])  # center stored in matrix convention
        frame = cv2.drawMarker(
            frame,
            (u, v),
            (0, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=5,
            line_type=cv2.LINE_AA,
        )
    cv2.imshow("hv_center", frame[:, :, ::-1])
    cv2.waitKey(1)
    return frame


@hydra.main(config_path="../../config", config_name="cfg_affordance")
def main(cfg):
    val = VREnvData(split="validation", log=None, **cfg.dataset)
    val_loader = DataLoader(val, num_workers=1, batch_size=1, pin_memory=True)
    print("val minibatches {}".format(len(val_loader)))
    from vapo.affordance.hough_voting import hough_voting as hv

    hv = hv.HoughVoting(**cfg.model_cfg.hough_voting)

    cm = plt.get_cmap("jet")
    colors = cm(np.linspace(0, 1, val.n_classes))
    for b_idx, b in enumerate(val_loader):
        # RGB
        frame, labels = b
        directions = labels["center_dirs"][0].detach().cpu().numpy()
        mask = labels["affordance"].detach().cpu().numpy()

        frame = frame[0].detach().cpu().numpy()
        # Undo normalization
        frame = ((frame + 1) * 255 / 2).astype("uint8")
        frame = np.transpose(frame, (1, 2, 0))
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        directions = np.transpose(directions, (1, 2, 0))
        flow_img = flowlib.flow_to_image(directions)  # RGB

        out_img = frame
        for label in range(1, val.n_classes):
            color = colors[label - 1]
            color[-1] = 0.3
            color = tuple((color * 255).astype("int32"))
            class_mask = np.zeros(mask.shape[1:])
            class_mask[mask[0] == label] = 255
            out_img = overlay_mask(class_mask, out_img, color)
        cv2.imshow("masks", out_img[:, :, ::-1])

        fg_mask = mask.any(axis=0).astype("uint8")  # Binary (0,1)
        fg_mask = torch.tensor(fg_mask).unsqueeze(0)
        mask = np.transpose(mask, (1, 2, 0)) * 255
        out_img = overlay_flow(flow_img, frame, mask)

        out_img = test_dir_labels(hv, out_img, fg_mask, labels["center_dirs"])
        out_img = cv2.resize(out_img, (200, 200), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("img", out_img[:, :, ::-1])
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
