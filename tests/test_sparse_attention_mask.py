import pytest
import torch
from torch import nn

from model.showui.modeling_showui import Qwen2VLDecoderLayer
from model.showui.utils import select_attention_mask


def test_projects_query_and_key_axes_for_prefill_mask():
    sequence_length = 5
    mask = torch.arange(sequence_length * sequence_length).reshape(
        1, 1, sequence_length, sequence_length
    )
    retain_mask = torch.tensor([True, False, True, False, True])

    projected = select_attention_mask(mask, retain_mask)
    indices = retain_mask.nonzero(as_tuple=True)[0]
    expected = mask.index_select(-2, indices).index_select(-1, indices)

    assert projected.shape == (1, 1, 3, 3)
    assert torch.equal(projected, expected)


def test_preserves_cached_key_columns_when_only_queries_match():
    sequence_length = 5
    mask = torch.arange(sequence_length * 8).reshape(1, 1, sequence_length, 8)
    retain_mask = torch.tensor([True, False, True, False, True])

    projected = select_attention_mask(mask, retain_mask, compact_key_axis=False)
    indices = retain_mask.nonzero(as_tuple=True)[0]

    assert projected.shape == (1, 1, 3, 8)
    assert torch.equal(projected, mask.index_select(-2, indices))


def test_compacts_dynamic_cache_prefix_and_current_keys():
    sequence_length = 5
    mask = torch.arange(sequence_length * 10).reshape(1, 1, sequence_length, 10)
    retain_mask = torch.tensor([True, False, True, False, True])

    projected = select_attention_mask(mask, retain_mask, past_length=2)
    query_indices = retain_mask.nonzero(as_tuple=True)[0]
    key_indices = torch.tensor([0, 1, 2, 4, 6])
    expected = mask.index_select(-2, query_indices).index_select(-1, key_indices)

    assert projected.shape == (1, 1, 3, 5)
    assert torch.equal(projected, expected)


def test_projects_flash_attention_padding_mask():
    mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]])
    retain_mask = torch.tensor([True, False, True, False, True])

    projected = select_attention_mask(mask, retain_mask)

    assert torch.equal(projected, torch.tensor([[1, 0, 0], [1, 1, 1]]))


def test_accepts_explicit_indices_and_requires_source_length():
    mask = torch.arange(25).reshape(1, 1, 5, 5)
    indices = torch.tensor([0, 3, 4])

    projected = select_attention_mask(mask, indices, sequence_length=5)
    expected = mask.index_select(-2, indices).index_select(-1, indices)

    assert torch.equal(projected, expected)
    with pytest.raises(ValueError, match="sequence_length"):
        select_attention_mask(mask, indices)


def test_rejects_empty_selection():
    with pytest.raises(ValueError, match="at least one token"):
        select_attention_mask(torch.ones(1, 1, 4, 4), torch.zeros(4, dtype=torch.bool))


def test_ui_guided_layer_projects_mask_and_preserves_batch_outputs(monkeypatch):
    layer = object.__new__(Qwen2VLDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_skip_ratio = 0.5
    layer.layer_skip_rand = False

    calls = {}

    def fake_attention(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        calls["hidden_states"] = hidden_states
        calls["attention_mask"] = attention_mask
        result = (hidden_states + 1.0,)
        if output_attentions:
            result += (torch.tensor(2),)
        if use_cache:
            result += (torch.tensor(3),)
        return result

    monkeypatch.setattr(Qwen2VLDecoderLayer, "navie_forward", fake_attention)

    batch_size, sequence_length, hidden_size = 2, 5, 4
    hidden_states = torch.zeros(batch_size, sequence_length, hidden_size)
    position_ids = torch.zeros(3, batch_size, sequence_length, dtype=torch.long)
    cache_position = torch.arange(sequence_length)
    position_embeddings = (
        torch.ones(3, batch_size, sequence_length, hidden_size),
        torch.zeros(3, batch_size, sequence_length, hidden_size),
    )
    attention_mask = torch.arange(sequence_length * (sequence_length + 1)).reshape(
        1, 1, sequence_length, sequence_length + 1
    )
    retain_mask = torch.tensor(
        [[True, False, True, False, True], [True, False, True, False, True]]
    )

    outputs = layer.ui_guide_forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        patch_pos=torch.zeros(batch_size, sequence_length, dtype=torch.long),
        select_mask=retain_mask,
        output_attentions=True,
        use_cache=True,
    )

    indices = retain_mask[0].nonzero(as_tuple=True)[0]
    expected_mask = attention_mask.index_select(-2, indices).index_select(-1, indices)
    assert torch.equal(calls["attention_mask"], expected_mask)
    assert calls["hidden_states"].shape == (batch_size, indices.numel(), hidden_size)
    assert torch.equal(
        outputs[0][:, indices], torch.ones(batch_size, indices.numel(), hidden_size)
    )
    assert torch.equal(
        outputs[0][:, ~retain_mask[0]], torch.zeros(batch_size, 2, hidden_size)
    )
    assert outputs[1].item() == 2
    assert outputs[2].item() == 3
