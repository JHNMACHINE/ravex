import torch
from torch.utils.data import BatchSampler, RandomSampler

from ravex._sampler import TrackedSampler

SAMPLES = 20
BATCH = 4
BATCHES_PER_EPOCH = SAMPLES // BATCH


def make_batch_sampler(seed=1234):
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(range(SAMPLES), generator=generator)
    return BatchSampler(sampler, batch_size=BATCH, drop_last=False)


def consume(tracked, iterator, count):
    """Take `count` batches the way the dataloader wrapper does.

    ``note_consumed`` is what the training loop's batch counter looks like from
    the sampler's side; the sampler's own yields run ahead of it whenever
    workers prefetch.
    """
    taken = []
    for _ in range(count):
        taken.append(list(next(iterator)))
        tracked.note_consumed()
    return taken


def epochs(tracked, count):
    result = []
    for _ in range(count):
        iterator = iter(tracked)
        batches = []
        for batch in iterator:
            batches.append(list(batch))
            tracked.note_consumed()
        result.append(batches)
    return result


def test_position_follows_the_training_loop():
    tracked = TrackedSampler(make_batch_sampler())
    iterator = iter(tracked)

    assert tracked.position == 0
    consume(tracked, iterator, 1)
    assert tracked.position == 1
    consume(tracked, iterator, 1)
    assert tracked.position == 2


def test_shuffled_order_is_reproducible_from_a_seed():
    first = epochs(TrackedSampler(make_batch_sampler()), 2)
    second = epochs(TrackedSampler(make_batch_sampler()), 2)
    assert first == second
    assert first[0] != first[1], "consecutive epochs must reshuffle"


def test_an_unseeded_sampler_is_made_replayable():
    # Without a generator, RandomSampler draws a fresh seed from the global RNG
    # on every epoch and no restart can reproduce the order.
    torch.manual_seed(7)
    plain = BatchSampler(RandomSampler(range(SAMPLES)), batch_size=BATCH, drop_last=False)
    assert plain.sampler.generator is None

    tracked = TrackedSampler(plain)
    assert plain.sampler.generator is not None, "Ravex must attach one"

    state = tracked.state()
    order = epochs(tracked, 1)

    replayed = TrackedSampler(
        BatchSampler(RandomSampler(range(SAMPLES)), batch_size=BATCH, drop_last=False)
    )
    replayed.restore(state)
    # The generator state travels in the checkpoint, so the order comes back
    # even though the second sampler was seeded from a different global RNG.
    assert epochs(replayed, 1) == order


def test_resume_mid_epoch_continues_the_same_order():
    reference = epochs(TrackedSampler(make_batch_sampler()), 1)[0]

    interrupted = TrackedSampler(make_batch_sampler())
    iterator = iter(interrupted)
    consumed = consume(interrupted, iterator, 3)
    state = interrupted.state()
    del iterator  # the process died here

    assert consumed == reference[:3]

    resumed = TrackedSampler(make_batch_sampler())
    resumed.restore(state)

    assert epochs(resumed, 1)[0] == reference[3:], "must pick up at batch 4"


def test_resume_on_an_epoch_boundary_replays_into_the_next_epoch():
    reference = epochs(TrackedSampler(make_batch_sampler()), 2)

    interrupted = TrackedSampler(make_batch_sampler())
    iterator = iter(interrupted)
    consume(interrupted, iterator, BATCHES_PER_EPOCH)
    # Checkpoint taken after the last batch of the epoch, before the loop
    # noticed the epoch was over.
    state = interrupted.state()
    assert state["position"] == BATCHES_PER_EPOCH

    resumed = TrackedSampler(make_batch_sampler())
    resumed.restore(state)
    replayed = epochs(resumed, 2)

    assert replayed[0] == [], "the finished epoch is skipped, not replayed"
    assert replayed[1] == reference[1], "training carries on with epoch 2"


def test_a_shorter_dataset_does_not_break_resume():
    # The user shrank the dataset between runs: skipping runs off the end.
    state = {"position": 999, "epoch_index": 0, "length": 999}
    resumed = TrackedSampler(make_batch_sampler())
    resumed.restore(state)

    replayed = epochs(resumed, 2)
    assert replayed[0] == []  # drained, with a warning
    assert len(replayed[1]) == BATCHES_PER_EPOCH  # and back to normal after


def test_attributes_pass_through_to_the_wrapped_sampler():
    original = make_batch_sampler()
    tracked = TrackedSampler(original)

    assert len(tracked) == len(original)
    assert tracked.batch_size == BATCH
    assert tracked.drop_last is False
    assert tracked.original is original
