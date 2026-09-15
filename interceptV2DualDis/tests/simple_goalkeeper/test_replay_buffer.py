"""Regression tests for beyondAMP's ReplayBuffer (rsl_rl_amp/storage/
replay_buffer.py) -- no direct test existed for this class before, despite
it being central to the 2026-09-15 persistent-buffer AMP fix. Written as
part of that investigation's independent verification pass: the class's own
docstring makes specific claims (same permutation/chunks replayed unchanged
across num_learning_epochs passes, FIFO wraparound correctness) that had
only been verified by reading the code, not by running it.
"""
import numpy as np
import torch

from rsl_rl_amp.storage.replay_buffer import ReplayBuffer


def test_feed_forward_generator_reuses_same_chunks_across_epochs():
    """Docstring claim: 'permute this region's pool ONCE per call ...
    replay the SAME chunks unchanged across all num_learning_epochs passes'.
    If chunks were instead reshuffled per-epoch, each of the 5 epochs would
    draw an independent random subset -- silently reverting the deterministic-
    reuse fix this mechanism exists for."""
    buf = ReplayBuffer(obs_dim=1, buffer_size=100, device="cpu")
    # Use the raw state value itself as an identity marker (0..num_samples-1)
    # so we can tell exactly which original rows each epoch's chunk contains.
    n = 40
    states = torch.arange(n, dtype=torch.float32).unsqueeze(-1)
    buf.insert(states, states.clone())

    num_mini_batches = 4
    num_learning_epochs = 5
    gen = buf.feed_forward_generator(num_mini_batches, num_learning_epochs)

    epochs_batches = []
    for _ in range(num_learning_epochs):
        epoch_batches = [next(gen)[0].squeeze(-1) for _ in range(num_mini_batches)]
        epochs_batches.append(epoch_batches)

    first_epoch = epochs_batches[0]
    for epoch_idx, epoch in enumerate(epochs_batches[1:], start=1):
        for batch_idx, (b0, bi) in enumerate(zip(first_epoch, epoch)):
            assert torch.equal(torch.sort(b0).values, torch.sort(bi).values), (
                f"epoch {epoch_idx}, minibatch {batch_idx} drew a DIFFERENT "
                f"set of samples than epoch 0 -- chunks are being reshuffled "
                f"per-epoch, not replayed unchanged as the docstring claims. "
                f"epoch0={torch.sort(b0).values.tolist()} "
                f"epoch{epoch_idx}={torch.sort(bi).values.tolist()}"
            )


def test_feed_forward_generator_no_replacement_within_a_chunk():
    """Each minibatch chunk should be sampled WITHOUT replacement -- the
    permutation is a real permutation (no repeated indices) sliced into
    contiguous chunks."""
    buf = ReplayBuffer(obs_dim=1, buffer_size=100, device="cpu")
    n = 40
    states = torch.arange(n, dtype=torch.float32).unsqueeze(-1)
    buf.insert(states, states.clone())

    gen = buf.feed_forward_generator(num_mini_batches=4, num_learning_epochs=1)
    seen = []
    for _ in range(4):
        batch, _ = next(gen)
        vals = batch.squeeze(-1).tolist()
        assert len(vals) == len(set(vals)), f"chunk has duplicate samples: {vals}"
        seen.extend(vals)
    assert len(seen) == len(set(seen)), "different chunks in the same call overlapped"


def test_insert_fifo_wraparound_preserves_newest_data():
    """insert() wraps around a fixed-size circular buffer. After wrapping,
    the buffer must hold the NEWEST buffer_size samples, not a corrupted mix."""
    buf = ReplayBuffer(obs_dim=1, buffer_size=10, device="cpu")

    first = torch.arange(0, 10, dtype=torch.float32).unsqueeze(-1)
    buf.insert(first, first.clone())
    assert buf.num_samples == 10

    # Insert 4 more -- should evict the 4 oldest (0,1,2,3), keep 4..9 + 10..13.
    second = torch.arange(10, 14, dtype=torch.float32).unsqueeze(-1)
    buf.insert(second, second.clone())
    assert buf.num_samples == 10

    remaining = set(buf.states[:10].squeeze(-1).tolist())
    expected = set(range(4, 14))
    assert remaining == expected, f"expected newest 10 samples {expected}, got {remaining}"


def test_num_samples_never_exceeds_buffer_size_across_many_inserts():
    buf = ReplayBuffer(obs_dim=1, buffer_size=50, device="cpu")
    for i in range(20):
        chunk = torch.full((7, 1), float(i))
        buf.insert(chunk, chunk.clone())
        assert buf.num_samples <= 50
    assert buf.num_samples == 50
