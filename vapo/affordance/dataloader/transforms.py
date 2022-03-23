import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T


class ScaleImageTensor(object):
    """Scale tensor of shape (batch, C, H, W) containing images to [0, 255] range
    Args:
        tensor (torch.tensor): Tensor to be scaled.
    Returns:
        Tensor: Scaled tensor.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor.float().div(255)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, clip=None):
        self.std = torch.Tensor(std)
        self.mean = torch.Tensor(mean)
        self.clip = clip

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        device = tensor.device
        _std = self.std.to(device)
        _mean = self.mean.to(device)

        t = tensor + torch.randn(tensor.size(), device=device) * _std + _mean
        if self.clip:
            t.clamp(self.clip[0], self.clip[1])
        return t

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class ThresholdMasks(object):
    def __init__(self, threshold):
        # Mask is between 0 and 255
        self.threshold = threshold

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Convert to 0-1
        assert isinstance(tensor, torch.Tensor)
        return (tensor > self.threshold).long()


class NormalizeVector(object):
    """Normalize a tensor vector with mean and standard deviation."""

    def __init__(self, mean=0.0, std=1.0):
        self.std = torch.Tensor(std)
        self.std[self.std == 0.0] = 1.0
        self.mean = torch.Tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class ColorTransform(object):
    def __init__(self, contrast=0.3, brightness=0.3, hue=0.3, prob=0.3):
        super().__init__()
        self.prob = prob
        self.jitter = T.ColorJitter(contrast=contrast, brightness=brightness, hue=hue)

    # Change image color
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        apply = np.random.rand() < self.prob
        tensor = self.jitter(tensor)
        if apply:
            tensor = self.jitter(tensor)
        return tensor


class RandomCrop(object):
    def __init__(self, size: int, rand_crop: float) -> None:
        super().__init__()
        _size = int(size * rand_crop)
        self.orig_size = size
        self.crop = torchvision.transforms.RandomCrop(_size)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        tensor = self.crop(tensor)
        return tensor


class DistanceTransform(object):
    """Apply distance transform to binary mask (0, 1)
    mask.shape = [C, W, H]
    mask.max() = 1
    mask.min() = 0
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        tensor = tensor.permute((1, 2, 0))  # C, W, H -> W, H, C
        mask = tensor.detach().cpu().numpy().astype(np.uint8)
        # cv2.imshow("in", mask*255)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        dist_np = np.array(dist)
        std_g = 2
        gauss_im = 1 / (std_g * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((1 - dist_np) / std_g) ** 2)
        gauss_im = cv2.normalize(gauss_im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask = (gauss_im * 255).astype(np.uint8)  # H, W
        # cv2.imshow("out", mask)
        # cv2.waitKey(1)
        mask = torch.from_numpy(mask).unsqueeze(0)  # 1, H, W
        return mask


# def prepare_input(self, image, bbox):
#     """Prepare the input for the network"""
#     image = image.astype(np.uint8)

#     rect_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
#     cv2.rectangle(rect_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
#                   (255, 255, 255), cv2.FILLED)
#     dist = cv2.distanceTransform(rect_mask, cv2.DIST_L2, 5)
#     dist = cv2.normalize(dist, None, alpha=0, beta=1,
#                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     dist_np = np.array(dist)
#     std_g = 2
#     gauss_im = 1/(std_g*np.sqrt(2*np.pi)) * \
#         np.exp(-0.5*((1-dist_np)/std_g)**2)
#     gauss_im = cv2.normalize(gauss_im, None, alpha=0, beta=1,
#                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     mask = (gauss_im * 255).astype(np.uint8)

#     comb = np.dstack((image, mask))
#     out = cv2.resize(comb, (1280, 720))

#     image = TF.to_pil_image(out[:, :, 0:3])
#     mask = TF.to_pil_image(np.expand_dims(out[:, :, 3], axis=2))

#     image = TF.to_tensor(image)
#     image = TF.normalize(image, BGR_MEAN, BGR_STD)
#     mask = TF.to_tensor(mask)

#     out = torch.cat((image, mask), dim=0)
#     return out
