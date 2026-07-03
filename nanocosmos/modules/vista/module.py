"""Vista3D Lightning module for volumetric connectomics segmentation."""

from nanocosmos.losses import AffinityFGLoss
from nanocosmos.models.vista import Vista3DWrapper
from nanocosmos.modules.vista.base import BaseVistaModule


class Vista3DModule(BaseVistaModule):
    """Vista3D volumetric segmentation module.

    Emits the single ``[B, HEAD_CHANNELS, D, H, W]`` affinity + sem + raw
    head (raw logits / linear values) supervised by ``AffinityFGLoss``.

    Checkpoint note: unlike the Cosmos modules, the Vista base does NOT call
    ``save_hyperparameters()``, so Vista checkpoints record no
    ``hyper_parameters`` provenance. The backbone class is config-conditional
    (SegResNetDS2 vs SegResNet, chosen by ``encoder_name`` / MONAI
    availability) and its ``model.backbone.*`` key set depends on
    ``feature_size``, so supply the same ``encoder_name`` / ``feature_size`` at
    load time as at train time -- they are not persisted in the checkpoint.
    """

    _SPATIAL_DIMS = 3
    _model_cls = Vista3DWrapper
    _loss_cls = AffinityFGLoss
