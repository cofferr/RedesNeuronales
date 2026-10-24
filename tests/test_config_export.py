import pandas as pd

from python import config


def test_config_csv_matches_config_py():
    exported = dict(pd.read_csv(config.CONFIG_CSV).itertuples(index=False, name=None))
    assert exported["learning_rate"] == config.LEARNING_RATE
    assert exported["epochs"] == config.EPOCHS
    assert exported["n_hidden"] == config.N_HIDDEN
    assert exported["n_train"] == config.N_TRAIN
