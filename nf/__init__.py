"""NF branch: multi-suite Mgas -> (Omega_m, sigma_8) parameter inference."""

from .encoder import ResNet3D
from .flow import ConditionalFlow
from .module import LitNFRegressor, MultiSuiteNFDataModule, MultiSuiteMgasDataset
from .predict import load_nf, predict_with_uncertainty

__all__ = [
    "ResNet3D", "ConditionalFlow", "LitNFRegressor",
    "MultiSuiteNFDataModule", "MultiSuiteMgasDataset",
    "load_nf", "predict_with_uncertainty",
]
