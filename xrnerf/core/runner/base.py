from mmcv.runner import EpochBasedRunner, IterBasedRunner


class NerfTrainRunner(IterBasedRunner):
    """NerfTrainRunner."""
    pass


class NerfTestRunner(EpochBasedRunner):
    """NerfTestRunner."""
    pass
