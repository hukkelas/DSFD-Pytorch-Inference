import setuptools
import torch
import torchvision

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 6], "Requires PyTorch >= 1.6"
torchvision_ver = [int(x) for x in torchvision.__version__.split(".")[:2]]
assert torchvision_ver >= [0, 3], "Requires torchvision >= 0.3"

setuptools.setup(
    name="face_detection",
    version="0.2.1",
    author="Håkon Hukkelås",
    description="A simple and lightweight package for state of the art face detection with GPU support.",
    long_description="".join(open("README.md", "r").readlines()),
    long_description_content_type="text/markdown",
    url="https://github.com/hukkelas/DSFD-Pytorch-Inference",
    python_requires='>=3.6',
    license="apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
    ],
    packages=setuptools.find_packages()
)
