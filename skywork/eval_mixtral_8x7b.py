from mmengine.config import read_base

with read_base():
    from ..configs.datasets.bbh.bbh_gen import bbh_datasets
    from ..configs.models.mixtral.mixtral_8x7b_32k import models
    from ..configs.summarizers.leaderboard import summarizer

datasets = bbh_datasets
