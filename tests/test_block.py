import pytest

from vllmlite.block import BlockAllocator, OutOfBlocks, hash_block


def test_alloc_and_free_roundtrip():
    a = BlockAllocator(4, 16, enable_prefix_caching=False)
    ids = [a.allocate()[0] for _ in range(4)]
    assert len(set(ids)) == 4
    assert a.num_free == 0
    with pytest.raises(OutOfBlocks):
        a.allocate()
    for i in ids:
        a.free(i)
    assert a.num_free == 4


def test_refcount_holds_block():
    a = BlockAllocator(2, 16, enable_prefix_caching=False)
    bid, _ = a.allocate()
    a.fork(bid)
    assert a.is_shared(bid)
    a.free(bid)
    assert not a.is_shared(bid)
    assert a.num_free == 1
    a.free(bid)
    assert a.num_free == 2


def test_double_free_raises():
    a = BlockAllocator(1, 16)
    bid, _ = a.allocate()
    a.free(bid)
    with pytest.raises(ValueError):
        a.free(bid)


def test_full_block_is_shareable_while_still_in_use():
    a = BlockAllocator(4, 4)
    bid, hit = a.allocate()
    assert not hit
    h = hash_block(None, (1, 2, 3, 4))
    a.mark_full(bid, h)
    # second request with the same prefix should land on the same block even
    # though the first one hasn't finished
    same, hit2 = a.allocate(h)
    assert hit2 and same == bid
    assert a.blocks[bid].ref_count == 2


def test_cached_block_is_evictable_once_unreferenced():
    a = BlockAllocator(2, 4)
    b1, _ = a.allocate()
    a.mark_full(b1, hash_block(None, (1, 2, 3, 4)))
    a.free(b1)
    assert a.num_free == 2
    # allocating fresh blocks should be willing to reclaim it
    x, _ = a.allocate()
    y, _ = a.allocate()
    assert {x, y} == {0, 1}


def test_hash_chains_the_prefix():
    h1 = hash_block(None, (1, 2))
    h2a = hash_block(h1, (3, 4))
    h2b = hash_block(hash_block(None, (9, 9)), (3, 4))
    # same block contents, different history, must not collide
    assert h2a != h2b


def test_hit_rate_tracking():
    a = BlockAllocator(4, 4)
    h = hash_block(None, (1, 2, 3, 4))
    bid, _ = a.allocate(h)
    a.mark_full(bid, h)
    a.allocate(h)
    assert a.stats()["prefix_hit_rate"] == 0.5
