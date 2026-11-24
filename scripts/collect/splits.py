"""Train/test seed split for oracle demonstration collection.

Per task (every roboclock task, one seed per episode):

- train: seeds 0-99, the first 50 at easy and the next 50 at medium. These are the seeds
  whose oracle demonstrations are recorded.
- train_spare: seeds 100-199, the same easy/medium halves. A spare is used only when the oracle
  fails a train seed (the failed rollout is not a usable demonstration); spares are taken in
  order within the same difficulty, so a task's demonstrations stay 50 easy + 50 medium.
- test: seeds 1000-1099. Collection never runs them. They are disjoint from every train and
  spare seed, so no evaluation episode shares a scene sample with a demonstration.

Training never includes hard. The test seeds carry no difficulty; evaluate them at whichever
levels the experiment needs (easy/medium are in-distribution, hard is not).
"""

TRAIN_PER_DIFFICULTY = 50
TRAIN_DIFFICULTIES = ("easy", "medium")
TRAIN_START, SPARE_START, TEST_START = 0, 100, 1000
TEST_COUNT = 100


def train_seeds(difficulty):
    index = TRAIN_DIFFICULTIES.index(difficulty)
    start = TRAIN_START + index * TRAIN_PER_DIFFICULTY
    return list(range(start, start + TRAIN_PER_DIFFICULTY))


def spare_seeds(difficulty):
    index = TRAIN_DIFFICULTIES.index(difficulty)
    start = SPARE_START + index * TRAIN_PER_DIFFICULTY
    return list(range(start, start + TRAIN_PER_DIFFICULTY))


def test_seeds():
    return list(range(TEST_START, TEST_START + TEST_COUNT))


def split_table(tasks):
    """{task_dir: {...}} for every (environment, task_id) in `tasks`."""
    test = set(test_seeds())
    table = {}
    for environment, task_id in tasks:
        entry = {
            "environment": environment,
            "task_id": task_id,
            "train": {d: train_seeds(d) for d in TRAIN_DIFFICULTIES},
            "train_spare": {d: spare_seeds(d) for d in TRAIN_DIFFICULTIES},
            "test": test_seeds(),
        }
        used = {s for d in TRAIN_DIFFICULTIES for s in entry["train"][d] + entry["train_spare"][d]}
        assert not used & test and len(used) == 4 * TRAIN_PER_DIFFICULTY
        table[task_dir(environment, task_id)] = entry
    return table


def task_dir(environment, task_id):
    return f"{environment}_{task_id}"
