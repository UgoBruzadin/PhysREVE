from setuptools import setup, find_packages

setup(
    name="physreve",
    version="0.4.0",
    description="Physics-Informed REVE Pretraining for EEG",
    author="Ugo Bruzadin Nunes",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.12",
        "numpy>=1.23",
        "scipy>=1.9",
        "scikit-learn>=1.1",
        "mne>=1.6",
        "moabb>=1.0",
        "tqdm>=4.64",
        "requests",
        "matplotlib",
        "seaborn",
    ],
    extras_require={
        "baselines": ["xgboost>=1.7"],
        "full": ["xgboost>=1.7", "einops"],
    },
)
