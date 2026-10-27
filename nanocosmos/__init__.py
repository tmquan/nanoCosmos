"""
Nanocosmos: a PyTorch Lightning infrastructure for connectomics
segmentation.  The model emits an affinity + foreground head; instances
are agglomerated at eval and inference by the Mutex Watershed.

Provides:
- MONAI-compatible dataset classes with a standardised interface
- Preprocessors for common data formats (TIFF, HDF5, NRRD, NIfTI)
- Cosmos 3 model wrappers (DiT + VAE backbone) for volumetric
  segmentation. Nano (16B), Edge (4B), and Super (64B) are the backbones.
- An affinity + foreground head supervised by ``AffinityFGLoss`` and
  agglomerated into instances at eval/inference by the Mutex Watershed
  (``nanocosmos.inference.mutex_watershed``).
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="The cuda.cudart module is deprecated",
    category=FutureWarning,
)

__version__ = "0.1.0"

from nanocosmos.datasets import (
    CircuitDataset,
    LazyVolDataset,
    SNEMI3DDataset,
    MICRONSDataset,
    FLYEM3DDataset,
    FLYWIREDataset,
    CREMI3DDataset,
    NeuronsDataset,
)
from nanocosmos.preprocessors import (
    BasePreprocessor,
    TIFFPreprocessor,
    HDF5Preprocessor,
    NRRDPreprocessor,
    NFTYPreprocessor,
)
from nanocosmos.datamodules import (
    CircuitDataModule,
    SNEMI3DDataModule,
    MICRONSDataModule,
    FLYEM3DDataModule,
    FLYWIREDataModule,
    CREMI3DDataModule,
    NeuronsDataModule,
)
from nanocosmos.losses import AffinityFGLoss, HEAD_CHANNELS, HEAD_LAYOUT, slice_head
from nanocosmos.models import (
    BaseModel,
    Cosmos3EdgeWrapper,
    Cosmos3Nano3DWrapper,
    Cosmos3SuperWrapper,
)
from nanocosmos.modules import (
    BaseCircuitModule,
    BaseCosmosModule,
    Cosmos3EdgeModule,
    Cosmos3Nano3DModule,
    Cosmos3SuperModule,
)

__all__ = [
    # Data
    "CircuitDataset",
    "LazyVolDataset",
    "SNEMI3DDataset",
    "MICRONSDataset",
    "FLYEM3DDataset",
    "FLYWIREDataset",
    "CREMI3DDataset",
    "NeuronsDataset",
    "BasePreprocessor",
    "TIFFPreprocessor",
    "HDF5Preprocessor",
    "NRRDPreprocessor",
    "NFTYPreprocessor",
    "CircuitDataModule",
    "SNEMI3DDataModule",
    "MICRONSDataModule",
    "FLYEM3DDataModule",
    "FLYWIREDataModule",
    "CREMI3DDataModule",
    "NeuronsDataModule",
    # Losses
    "AffinityFGLoss",
    "HEAD_CHANNELS",
    "HEAD_LAYOUT",
    "slice_head",
    # Models (backbone wrappers)
    "BaseModel",
    "Cosmos3Nano3DWrapper",
    "Cosmos3EdgeWrapper",
    "Cosmos3SuperWrapper",
    # Modules (Lightning)
    "BaseCircuitModule",
    "BaseCosmosModule",
    "Cosmos3Nano3DModule",
    "Cosmos3EdgeModule",
    "Cosmos3SuperModule",
]
