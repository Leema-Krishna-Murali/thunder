import torch

from thunder.models import pretrained_models


class _DummyModel:
    def __init__(self):
        self.pos_embed = None
        self.loaded = None

    def load_state_dict(self, checkpoint):
        self.loaded = checkpoint


def test_get_openmidnight_uses_pretrained_false(monkeypatch):
    captured = {}
    dummy_model = _DummyModel()
    pos_embed = torch.zeros((1, 1, 1), dtype=torch.float32)

    def _mock_hub_load(repo_or_dir, model, *args, **kwargs):
        captured["repo_or_dir"] = repo_or_dir
        captured["model"] = model
        captured["kwargs"] = kwargs
        return dummy_model

    def _mock_torch_load(path, map_location="cpu"):
        captured["ckpt_path"] = path
        return {"pos_embed": pos_embed}

    monkeypatch.setattr(pretrained_models.torch.hub, "load", _mock_hub_load)
    monkeypatch.setattr(pretrained_models.torch, "load", _mock_torch_load)

    model, transform = pretrained_models.get_openmidnight("/tmp/teacher_checkpoint.pt")

    assert model is dummy_model
    assert callable(transform)
    assert captured["repo_or_dir"] == "facebookresearch/dinov2"
    assert captured["model"] == "dinov2_vitg14_reg"
    assert captured["kwargs"].get("pretrained") is False
    assert captured["ckpt_path"] == "/tmp/teacher_checkpoint.pt"
    assert isinstance(dummy_model.pos_embed, torch.nn.Parameter)
