"""
SSD cache pool using safetensors format.
Stores compressed KV blocks with page pool management for fast load/unload.
"""

import json
import logging
import shutil
from pathlib import Path
from uuid import UUID

import numpy as np

from .kv_pager import PagedKVBlock

logger = logging.getLogger(__name__)

try:
    from safetensors.numpy import load_file, save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    logger.warning("safetensors not installed. Using NumPy fallback.")


class SSDPagePool:
    """
    Manages KV cache pages on NVMe SSD storage.

    On Cezanne UMA with NVMe:
    - Cold load (text prefill): 74,219ms for 16K context
    - Load compressed KV: ~795ms (93x speedup)
    """

    def __init__(self, cache_dir: str = "~/.cache/anvil/kv_pages",
                 max_gb: int = 30):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_gb * 1024**3
        self._index_path = self.cache_dir / "page_index.json"
        self._index: dict[str, dict] = {}
        self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Corrupt page index, starting fresh: %s", e)
                self._index = {}

    def _save_index(self):
        with open(self._index_path, 'w') as f:
            json.dump(self._index, f, indent=2)

    def save_block(self, block: PagedKVBlock) -> str:
        """Save compressed KV block to SSD."""
        block_id = str(block.block_id)
        page_path = self.cache_dir / f"{block_id}.safetensors"
        meta_path = self.cache_dir / f"{block_id}.meta.json"

        # Save metadata
        with open(meta_path, 'w') as f:
            json.dump({
                "block_id": block_id,
                "model_id": block.model_id,
                "agent_id": block.agent_id,
                "token_count": block.token_count,
                "context_hash": block.context_hash,
                "metadata": block.metadata,
                "key_shape": list(block.key_q8.shape) if block.key_q8 is not None else [],
                "value_shape": list(block.value_3bit.shape) if block.value_3bit is not None else [],
            }, f)

        # Save tensors
        tensors = {}
        if block.key_q8 is not None:
            tensors["key_q8"] = block.key_q8
        if block.value_3bit is not None:
            tensors["value_3bit"] = block.value_3bit

        if HAS_SAFETENSORS:
            save_file(tensors, str(page_path))
        else:
            np.savez(str(page_path.with_suffix('.npz')), **tensors)

        # Update index
        self._index[block_id] = {
            "agent_id": block.agent_id,
            "token_count": block.token_count,
            "page_path": str(page_path),
            "size_bytes": sum(t.nbytes for t in tensors.values()) if tensors else 0,
        }
        self._save_index()

        # Enforce max disk cache
        self._evict_if_needed()

        logger.info(f"KV block {block_id} saved ({block.token_count} tokens)")
        return block_id

    def load_block(self, block_id: str | UUID) -> PagedKVBlock | None:
        """Load compressed KV block from SSD."""
        block_id = str(block_id)
        if block_id not in self._index:
            logger.warning(f"Block {block_id} not found in index")
            return None

        page_path = self.cache_dir / f"{block_id}.safetensors"
        meta_path = self.cache_dir / f"{block_id}.meta.json"

        if not page_path.exists() and not page_path.with_suffix('.npz').exists():
            logger.warning(f"Block file not found: {page_path}")
            return None

        # Load metadata
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt block metadata for %s: %s", block_id, e)
            return None

        # Load tensors
        if HAS_SAFETENSORS and page_path.exists():
            data = load_file(str(page_path))
            key_q8 = data["key_q8"]
            value_3bit = data["value_3bit"]
        else:
            npz_path = page_path.with_suffix('.npz')
            data = np.load(str(npz_path))
            key_q8 = data["key_q8"]
            value_3bit = data["value_3bit"]

        block = PagedKVBlock(
            block_id=UUID(block_id),
            model_id=meta["model_id"],
            agent_id=meta["agent_id"],
            token_count=meta["token_count"],
            key_q8=key_q8,
            value_3bit=value_3bit,
            context_hash=meta["context_hash"],
            metadata=meta["metadata"],
        )
        return block

    def delete_block(self, block_id: str | UUID):
        """Remove block from SSD cache."""
        block_id = str(block_id)
        if block_id in self._index:
            for ext in ['.safetensors', '.npz', '.meta.json']:
                p = self.cache_dir / f"{block_id}{ext}"
                if p.exists():
                    p.unlink()
            del self._index[block_id]
            self._save_index()

    def _evict_if_needed(self):
        """LRU eviction when disk cache exceeds max_bytes."""
        total = sum(e["size_bytes"] for e in self._index.values())
        if total <= self.max_bytes:
            return

        # Simple LRU: evict oldest entries (by insertion order)
        to_evict = []
        for bid, entry in sorted(self._index.items()):
            if total <= self.max_bytes * 0.8:
                break
            to_evict.append(bid)
            total -= entry["size_bytes"]

        for bid in to_evict:
            self.delete_block(bid)
            logger.info(f"Evicted KV block {bid}")

    def list_blocks(self, agent_id: str | None = None) -> list[dict]:
        if agent_id:
            return [
                {"block_id": bid, **entry}
                for bid, entry in self._index.items()
                if entry["agent_id"] == agent_id
            ]
        return [{"block_id": bid, **entry} for bid, entry in self._index.items()]

    def clear(self):
        """Clear all cached blocks."""
        resolved = self.cache_dir.resolve()
        parent = resolved.parent
        if str(resolved) in ("/", str(Path.home()), str(Path.home().resolve())):
            raise PermissionError(
                f"Refusing to clear {resolved}: refusing to wipe system directory"
            )
        if parent == resolved:
            raise PermissionError(f"Refusing to clear {resolved}: no parent directory")
        shutil.rmtree(resolved)
        self.cache_dir.mkdir(parents=True)
        self._index = {}
        self._save_index()
