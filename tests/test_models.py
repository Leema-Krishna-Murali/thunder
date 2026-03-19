import os
import tempfile

import pytest


def test_download_models(temp_env_dir):
    """Tests the download and split generation of the ocelot dataset."""
    # Download the dataset
    from thunder import download_models

    download_models("dinov2base")

    # Check if the dataset is in the temp_env_dir
    model_path = os.path.join(temp_env_dir, "pretrained_ckpts/dinov2base")
    assert os.path.exists(
        model_path
    ), "Failed to download model (Model directory not found where expected)."


def test_get_openmidnight_loads_arch_without_hub_weights(monkeypatch):
    import torch

    from thunder.models import pretrained_models

    class DummyModel:
        def __init__(self):
            self.pos_embed = None
            self.loaded_state = None

        def load_state_dict(self, state_dict):
            self.loaded_state = state_dict

    dummy_model = DummyModel()
    hub_call = {}
    checkpoint = {"pos_embed": torch.randn(1, 257, 1536)}

    def fake_hub_load(repo, model_name, **kwargs):
        hub_call["repo"] = repo
        hub_call["model_name"] = model_name
        hub_call["kwargs"] = kwargs
        return dummy_model

    def fake_torch_load(path, map_location=None):
        assert path == "/tmp/openmidnight.ckpt"
        assert map_location == "cpu"
        return checkpoint

    monkeypatch.setattr(pretrained_models.torch.hub, "load", fake_hub_load)
    monkeypatch.setattr(pretrained_models.torch, "load", fake_torch_load)

    model, transform = pretrained_models.get_openmidnight("/tmp/openmidnight.ckpt")

    assert model is dummy_model
    assert hub_call["repo"] == "facebookresearch/dinov2"
    assert hub_call["model_name"] == "dinov2_vitg14_reg"
    assert hub_call["kwargs"] == {"pretrained": False}
    assert "weights" not in hub_call["kwargs"]
    assert isinstance(model.pos_embed, torch.nn.Parameter)
    assert torch.equal(model.loaded_state["pos_embed"], checkpoint["pos_embed"])
    assert callable(transform)
