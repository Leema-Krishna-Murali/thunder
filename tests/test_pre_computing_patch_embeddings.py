import os
from types import SimpleNamespace

import h5py
import torch

import thunder.tasks.pre_computing_patch_embeddings as precompute_module


class _DummyModel:
    def __init__(self, vlm: bool = False):
        self.vlm = vlm

    def get_transform(self):
        return lambda x: x

    def get_embeddings(
        self,
        src,
        model,
        task_type="linear_probing",
        text_aligned_im_emb=False,
        text_emb=False,
    ):
        if text_emb:
            return torch.ones((len(src), 4), dtype=torch.float32)
        return torch.ones((src.shape[0], 4), dtype=torch.float32)

    def to(self, device):
        return self

    def eval(self):
        return self


def _make_cfg(vlm: bool = False, id_to_classname=None):
    dataset_cfg = SimpleNamespace(div_patches=False)
    if id_to_classname is not None:
        dataset_cfg.id_to_classname = id_to_classname
    return SimpleNamespace(
        dataset=dataset_cfg,
        pretrained_model=SimpleNamespace(vlm=vlm),
        task=SimpleNamespace(pre_comp_emb_batch_size=2),
    )


def test_precompute_skips_missing_splits(monkeypatch, tmp_path):
    cfg = _make_cfg(vlm=False)

    monkeypatch.setattr(
        precompute_module,
        "get_data",
        lambda *args, **kwargs: {
            "train": {"images": ["a"], "labels": [0]},
            "val": {"images": ["b"], "labels": [1]},
        },
    )
    monkeypatch.setattr(precompute_module, "PatchDataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(precompute_module, "DataLoader", lambda *args, **kwargs: object())

    called_splits = []

    def _record_split(embeddings_folder, *args, **kwargs):
        called_splits.append(os.path.basename(embeddings_folder))

    monkeypatch.setattr(
        precompute_module, "pre_computing_patch_embeddings_split", _record_split
    )

    precompute_module.pre_computing_patch_embeddings(
        cfg=cfg,
        embeddings_folder=str(tmp_path / "embeddings"),
        device="cpu",
        dataset_name="custom_dataset",
        base_data_folder=str(tmp_path),
        data_compatible_tasks=["linear_probing"],
        adaptation_type="frozen",
        base_embeddings_folder=str(tmp_path),
        model_name="dummy",
        image_pre_loading=False,
        embedding_pre_loading=False,
        model_cls=_DummyModel(vlm=False),
    )

    assert called_splits == ["train", "val"]


def test_precompute_normalizes_string_class_ids(monkeypatch, tmp_path):
    cfg = _make_cfg(vlm=True, id_to_classname={"0": "normal", "1": "tumor"})

    monkeypatch.setattr(
        precompute_module,
        "get_data",
        lambda *args, **kwargs: {
            "train": {"images": ["a"], "labels": [0]},
            "val": {"images": ["b"], "labels": [1]},
            "test": {"images": ["c"], "labels": [1]},
        },
    )
    monkeypatch.setattr(precompute_module, "PatchDataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(precompute_module, "DataLoader", lambda *args, **kwargs: object())

    captured_id_to_classname = {}

    def _capture_id2classnames(embeddings_folder, *args, **kwargs):
        split_name = os.path.basename(embeddings_folder)
        captured_id_to_classname[split_name] = kwargs.get("id2classnames")

    monkeypatch.setattr(
        precompute_module,
        "pre_computing_patch_embeddings_split",
        _capture_id2classnames,
    )

    precompute_module.pre_computing_patch_embeddings(
        cfg=cfg,
        embeddings_folder=str(tmp_path / "embeddings"),
        device="cpu",
        dataset_name="custom_dataset",
        base_data_folder=str(tmp_path),
        data_compatible_tasks=["linear_probing"],
        adaptation_type="frozen",
        base_embeddings_folder=str(tmp_path),
        model_name="dummy",
        image_pre_loading=False,
        embedding_pre_loading=False,
        model_cls=_DummyModel(vlm=True),
    )

    assert captured_id_to_classname["train"] is None
    assert captured_id_to_classname["val"] is None
    assert captured_id_to_classname["test"] == {0: "normal", 1: "tumor"}


def test_text_embeddings_are_overwritten_without_failure(tmp_path):
    out_dir = str(tmp_path / "test_split")
    dataloader = [
        {
            "image": torch.ones((2, 3, 8, 8), dtype=torch.float32),
            "label": torch.tensor([0, 1], dtype=torch.int64),
        }
    ]

    def _extract_embedding(
        src,
        pretrained_model,
        task_type="linear_probing",
        text_aligned_im_emb=False,
        text_emb=False,
    ):
        if text_emb:
            return torch.ones((len(src), 4), dtype=torch.float32)
        return torch.ones((src.shape[0], 4), dtype=torch.float32)

    precompute_module.pre_computing_patch_embeddings_split(
        embeddings_folder=out_dir,
        dataloader=dataloader,
        pretrained_model=_DummyModel(vlm=True),
        extract_embedding=_extract_embedding,
        task_type="linear_probing",
        device="cpu",
        id2classnames={0: "normal", 1: "tumor"},
    )
    precompute_module.pre_computing_patch_embeddings_split(
        embeddings_folder=out_dir,
        dataloader=dataloader,
        pretrained_model=_DummyModel(vlm=True),
        extract_embedding=_extract_embedding,
        task_type="linear_probing",
        device="cpu",
        id2classnames={0: "normal", 1: "tumor"},
    )

    with h5py.File(os.path.join(out_dir, "embeddings.h5"), "r") as emb_h5:
        assert len(emb_h5.keys()) == 4
    with h5py.File(os.path.join(out_dir, "text_embeddings.h5"), "r") as text_h5:
        assert sorted(text_h5.keys()) == ["0", "1"]
