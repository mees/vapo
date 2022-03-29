from setuptools import setup

setup(
    name="vapo",
    version="1.0",
    description="Python Distribution Utilities",
    packages=["vapo"],
    install_requires=[
        "hydra-core",
        "opencv-python",
        "pytorch-lightning==1.5.9",
        "segmentation-models-pytorch",
        "pybullet",
        "numpy-quaternion",
        "hydra-colorlog",
        "pypng",
        "tqdm",
        "wandb",
        "pynput",
        "omegaconf",
        "flake8",
        "Pillow",
        "matplotlib",
    ],
)
