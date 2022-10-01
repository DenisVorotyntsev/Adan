from setuptools import setup, find_packages

setup(
    name="adan-tensorflow",
    packages=find_packages(exclude=[]),
    version="1.0.1",
    license="MIT",
    description="Adan - (ADAptive Nesterov momentum algorithm) Optimizer in Tensorflow",
    author="Denis Vorotyntsev",
    long_description_content_type="text/markdown",
    url="https://github.com/DenisVorotyntsev/adan",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "optimizer",
    ],
    install_requires=[
        """
    tensorflow>=2.3.0
    """,
    ],
)
