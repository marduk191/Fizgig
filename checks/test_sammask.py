"""SAM 3.1 masks (Image Prep's masks from words): the words typed, the downloaded model file, and the checkpoint's
key conversion - no model, no GPU."""
import torch

from fizgig.dataset import sammask
from fizgig.sam3.segment import _split_state_dict


def test_words_split_on_commas_and_lines():
    assert sammask.split_words(" person, hair ,\nthe dog,, ") == ["person", "hair", "the dog"]
    assert sammask.split_words("") == []


def test_model_is_the_downloaded_file_in_the_models_folder(tmp_path):
    assert sammask.find_model(str(tmp_path)) == ""
    (tmp_path / sammask.NAME).write_bytes(b"x")
    assert sammask.find_model(str(tmp_path)) == str(tmp_path / sammask.NAME)


def test_checkpoint_keys_split_into_model_and_text_encoder():
    enc = "detector.backbone.language_backbone.encoder."
    sd = {enc + "positional_embedding": torch.zeros(32, 4), enc + "token_embedding.weight": torch.zeros(10, 4),
          enc + "ln_final.weight": torch.zeros(4), enc + "ln_final.bias": torch.zeros(4),
          "detector.backbone.language_backbone.resizer.weight": torch.zeros(2, 4),
          "tracker.model.x.in_proj_weight": torch.arange(6.0)[:, None],
          "tracker.model.y.attn.freqs_cis": torch.zeros(1)}
    for i in range(24):
        r = f"{enc}transformer.resblocks.{i}."
        for n in ("ln_1", "ln_2", "mlp.c_fc", "mlp.c_proj", "attn.out_proj"):
            sd[r + n + ".weight"], sd[r + n + ".bias"] = torch.zeros(4), torch.zeros(4)
        sd[r + "attn.in_proj_weight"], sd[r + "attn.in_proj_bias"] = torch.arange(12.0)[:, None], torch.zeros(12)
    model, clip = _split_state_dict(sd)
    assert "detector.backbone.language_backbone.resizer.weight" in model             # the resizer stays in the model
    assert [k for k in model if k.startswith("tracker.x.")] == ["tracker.x.q_proj.weight", "tracker.x.k_proj.weight",
                                                                "tracker.x.v_proj.weight"]
    assert model["tracker.x.k_proj.weight"].flatten().tolist() == [2.0, 3.0]
    assert not any("freqs_cis" in k for k in model)
    assert clip["text_model.encoder.layers.3.self_attn.v_proj.weight"].flatten().tolist() == [8.0, 9.0, 10.0, 11.0]
    assert "text_model.final_layer_norm.weight" in clip and len(clip) == 4 + 24 * 16
