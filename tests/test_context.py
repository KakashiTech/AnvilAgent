from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from anvil.memory.context_restorer import ContextRestorer, ContextScheduler
from anvil.memory.kv_pager import AsymmetricKVQuantizer


@pytest.fixture
def quantizer():
    return AsymmetricKVQuantizer()


@pytest.fixture
def restorer(quantizer):
    return ContextRestorer(quantizer=quantizer)


@pytest.fixture
def sample_block(quantizer):
    n_heads, n_tokens, d_k = 8, 64, 128
    d_v = d_k
    rng = np.random.default_rng(42)
    keys = rng.normal(0, 1, (n_heads, n_tokens, d_k)).astype(np.float32)
    values = rng.normal(0, 1, (n_heads, n_tokens, d_v)).astype(np.float32)
    return quantizer.compress_block(keys, values, agent_id="test_agent", model_id="phi-4-mini-q4")


class TestKVFormatValidation:
    def test_valid_3d_tensors_passes(self, restorer):
        keys = np.zeros((8, 64, 128), dtype=np.float32)
        values = np.zeros((8, 64, 128), dtype=np.float32)
        assert restorer._validate_kv_format(keys, values) is True

    def test_2d_tensors_fail(self, restorer):
        keys = np.zeros((8, 128), dtype=np.float32)
        values = np.zeros((8, 128), dtype=np.float32)
        assert restorer._validate_kv_format(keys, values) is False

    def test_4d_tensors_fail(self, restorer):
        keys = np.zeros((1, 8, 64, 128), dtype=np.float32)
        values = np.zeros((1, 8, 64, 128), dtype=np.float32)
        assert restorer._validate_kv_format(keys, values) is False

    def test_head_mismatch_fails(self, restorer):
        keys = np.zeros((8, 64, 128), dtype=np.float32)
        values = np.zeros((4, 64, 128), dtype=np.float32)
        assert restorer._validate_kv_format(keys, values) is False

    def test_token_mismatch_fails(self, restorer):
        keys = np.zeros((8, 64, 128), dtype=np.float32)
        values = np.zeros((8, 32, 128), dtype=np.float32)
        assert restorer._validate_kv_format(keys, values) is False


class TestContextInjection:
    @pytest.mark.asyncio
    async def test_inject_valid_block_returns_true(self, restorer, sample_block):
        result = restorer.inject_context(sample_block, slot_id=0)
        assert result is True

    @pytest.mark.asyncio
    async def test_inject_sets_active_context(self, restorer, sample_block):
        restorer.inject_context(sample_block, slot_id=1)
        info = restorer.get_active_context_info()
        assert info is not None
        assert info["agent_id"] == "test_agent"
        assert info["slot_id"] == 1
        assert info["token_count"] == 64

    @pytest.mark.asyncio
    async def test_inject_bad_shapes_returns_false(self, restorer, quantizer):
        keys = np.zeros((4, 64, 128), dtype=np.float32)
        values = np.zeros((8, 64, 128), dtype=np.float32)
        bad_block = quantizer.compress_block(keys, values)
        result = restorer.inject_context(bad_block)
        assert result is False

    @pytest.mark.asyncio
    async def test_inject_bad_shapes_clears_context(self, restorer, quantizer):
        keys = np.zeros((4, 64, 128), dtype=np.float32)
        values = np.zeros((8, 64, 128), dtype=np.float32)
        bad_block = quantizer.compress_block(keys, values)
        restorer.inject_context(bad_block)
        assert restorer.get_active_context_info() is None

    @pytest.mark.asyncio
    async def test_inject_updates_slot_id(self, restorer, sample_block):
        restorer.inject_context(sample_block, slot_id=2)
        info = restorer.get_active_context_info()
        assert info["slot_id"] == 2


class TestAgentSlotRegistration:
    def test_register_new_agent_assigns_slot(self, restorer):
        scheduler = ContextScheduler(restorer)
        slot = scheduler.register_agent("agent_a")
        assert isinstance(slot, int)
        assert 0 <= slot < 4

    def test_register_agent_returns_same_slot(self, restorer):
        scheduler = ContextScheduler(restorer)
        slot1 = scheduler.register_agent("agent_b")
        slot2 = scheduler.register_agent("agent_b")
        assert slot1 == slot2

    def test_register_multiple_agents(self, restorer):
        scheduler = ContextScheduler(restorer)
        slots = {
            "agent_a": scheduler.register_agent("agent_a"),
            "agent_b": scheduler.register_agent("agent_b"),
            "agent_c": scheduler.register_agent("agent_c"),
        }
        assert len(set(slots.values())) == len(slots)

    def test_slot_wraparound(self, restorer):
        scheduler = ContextScheduler(restorer)
        scheduler._max_slots = 2
        slots = set()
        for i in range(5):
            slots.add(scheduler.register_agent(f"agent_{i}"))
        assert len(slots) == 2

    def test_get_slot_for_unknown_agent(self, restorer):
        scheduler = ContextScheduler(restorer)
        assert scheduler.get_slot_for("nonexistent") is None

    def test_get_slot_for_registered_agent(self, restorer):
        scheduler = ContextScheduler(restorer)
        slot = scheduler.register_agent("agent_x")
        assert scheduler.get_slot_for("agent_x") == slot


class TestContextSwitchCycle:
    @pytest.mark.asyncio
    async def test_switch_to_agent_with_block(self, restorer, sample_block):
        scheduler = ContextScheduler(restorer)
        result = await scheduler.switch_to_agent("switch_agent", block=sample_block)
        assert result is True

    @pytest.mark.asyncio
    async def test_switch_to_agent_without_block(self, restorer):
        scheduler = ContextScheduler(restorer)
        result = await scheduler.switch_to_agent("cold_agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_switch_registers_agent(self, restorer, sample_block):
        scheduler = ContextScheduler(restorer)
        await scheduler.switch_to_agent("new_agent", block=sample_block)
        assert scheduler.get_slot_for("new_agent") is not None

    @pytest.mark.asyncio
    async def test_switch_updates_context(self, restorer, sample_block):
        scheduler = ContextScheduler(restorer)
        await scheduler.switch_to_agent("ctx_test", block=sample_block)
        info = restorer.get_active_context_info()
        assert info is not None
        assert info["agent_id"] == sample_block.agent_id
        assert scheduler.get_slot_for("ctx_test") == 0

    @pytest.mark.asyncio
    async def test_switch_cold_then_hot(self, restorer, sample_block):
        scheduler = ContextScheduler(restorer)
        await scheduler.switch_to_agent("multi", block=None)
        assert restorer.get_active_context_info() is None
        await scheduler.switch_to_agent("multi", block=sample_block)
        info = restorer.get_active_context_info()
        assert info is not None
        assert info["agent_id"] == sample_block.agent_id


class TestActiveContextTracking:
    def test_no_active_context_returns_none(self, restorer):
        assert restorer.get_active_context_info() is None

    @pytest.mark.asyncio
    async def test_clear_context_clears_active(self, restorer, sample_block):
        restorer.inject_context(sample_block, slot_id=0)
        assert restorer.get_active_context_info() is not None
        restorer.clear_context(slot_id=0)
        assert restorer.get_active_context_info() is None

    @pytest.mark.asyncio
    async def test_clear_context_different_slot_does_not_clear(self, restorer, sample_block):
        restorer.inject_context(sample_block, slot_id=1)
        restorer.clear_context(slot_id=2)
        assert restorer.get_active_context_info() is not None

    @pytest.mark.asyncio
    async def test_reinject_replaces_context(self, restorer, quantizer):
        keys_a = np.zeros((4, 16, 64), dtype=np.float32)
        vals_a = np.zeros((4, 16, 64), dtype=np.float32)
        block_a = quantizer.compress_block(keys_a, vals_a, agent_id="agent_a")
        keys_b = np.ones((4, 32, 64), dtype=np.float32)
        vals_b = np.ones((4, 32, 64), dtype=np.float32)
        block_b = quantizer.compress_block(keys_b, vals_b, agent_id="agent_b")
        restorer.inject_context(block_a)
        assert restorer.get_active_context_info()["agent_id"] == "agent_a"
        restorer.inject_context(block_b)
        assert restorer.get_active_context_info()["agent_id"] == "agent_b"

    def test_format_for_llamacpp(self, restorer, sample_block, quantizer):
        keys, values = quantizer.decompress_block(sample_block)
        fmt = restorer._format_for_llamacpp(keys, values, sample_block)
        assert fmt["agent_id"] == "test_agent"
        assert fmt["n_tokens"] == 64
        assert fmt["key_shape"] == [8, 64, 128]
        assert isinstance(fmt["keys_mean"], float)
        assert isinstance(fmt["keys_std"], float)

    def test_block_id_in_active_context(self, restorer, sample_block):
        block_id = sample_block.block_id
        restorer._active_context = sample_block
        restorer._slot_id = 0
        info = restorer.get_active_context_info()
        assert info["block_id"] == str(block_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
