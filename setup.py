from setuptools import setup, find_packages

setup(
    name="rl-robot-control",
    version="0.1.0",
    description="基于深度强化学习的工业机器人智能控制系统",
    author="Your Name",
    author_email="youremail@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "gymnasium>=0.26.0",
        "pybullet>=3.2.1",
        "matplotlib>=3.4.0",
        "pyyaml>=6.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "tensorboard>=2.6.0",
        "tqdm>=4.62.0",
        "pytest>=6.2.5",
        "scikit-learn>=0.24.2"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 