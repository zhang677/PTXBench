"""Tinker SFT training used by the published PTXBench experiments."""

import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import chz
import datasets
import tinker
from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)
from tinker_cookbook.utils.lr_scheduling import LRSchedule

logger = logging.getLogger(__name__)


@chz.chz
class FromParquetFileBuilder(ChatDatasetBuilder):
    """Load chat conversations from a parquet file with a ``messages`` column.

    When ``filter_over_max_length`` is true, samples whose tokenized length exceeds
    ``common_config.max_length`` are dropped — truncating them would zero out
    assistant loss weights since user content precedes assistant in each row.
    """

    file_path: str
    test_size: int = 0
    shuffle_seed: int = 0
    filter_over_max_length: bool = True

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        ds = datasets.load_dataset("parquet", data_files=self.file_path, split="train")
        assert isinstance(ds, datasets.Dataset)
        if "messages" not in ds.column_names:
            raise ValueError(
                f"Parquet must contain a 'messages' column. Got: {ds.column_names}"
            )
        ds = ds.select_columns(["messages"])

        train_on_what = (
            renderers.TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        renderer = self.renderer
        max_length = self.common_config.max_length

        if self.filter_over_max_length and max_length is not None:
            before = len(ds)
            tokenizer = self.tokenizer

            def fits(row: dict) -> bool:
                n = sum(len(tokenizer.encode(m["content"])) for m in row["messages"])
                return n <= max_length

            ds = ds.filter(fits)
            logger.info(
                f"Filtered {before - len(ds)}/{before} samples exceeding "
                f"max_length={max_length}; kept {len(ds)}"
            )

        if self.shuffle_seed is not None:
            ds = ds.shuffle(seed=self.shuffle_seed)

        if self.test_size > 0 and len(ds) > self.test_size:
            test_ds = ds.take(self.test_size)
            train_ds = ds.skip(self.test_size)
        else:
            train_ds = ds
            test_ds = None

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(row["messages"], renderer, max_length, train_on_what)

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )
        test_dataset = (
            SupervisedDatasetFromHFDataset(test_ds, batch_size=len(test_ds), map_fn=map_fn)
            if test_ds is not None
            else None
        )
        return train_dataset, test_dataset


@chz.chz
class CLIConfig:
    # Data
    dataset_path: str = str(
        Path(os.environ.get("PTXBENCH_DATA_ROOT", "data"))
        / "sft_experiments"
        / "dataset.parquet"
    )
    test_size: int = 0
    shuffle_seed: int = 0

    # Model (Qwen3.5-35B-A3B; matches MODEL_PRESET=qwen35-35b-a3b)
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    renderer_name: str | None = None  # auto-resolved if None
    load_checkpoint_path: str | None = None

    # Training (mirrors launch.sh: LR=1e-7, batch=2, epoch=30).
    # Tinker caps per-request sequence length at 65536 for this model — samples
    # exceeding that are filtered out by FromParquetFileBuilder (220/344 fit).
    learning_rate: float = 1e-7
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 30
    batch_size: int = 2
    max_length: int = 65536
    # Every row has a single assistant turn, so LAST_ASSISTANT_MESSAGE is equivalent to
    # ALL_ASSISTANT_MESSAGES and avoids the qwen3_5 extension-property warning.
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE

    # LoRA (Tinker is LoRA-only; the source script ran full FT)
    lora_rank: int = 32

    # Checkpointing (mirrors SAVE_INTERVAL=50, SAVE_CHECKPOINTS=1)
    save_every: int = 50
    eval_every: int = 0
    infrequent_eval_every: int = 0

    # Logging (mirrors WANDB_PROJECT=qwen35-35b, RUN_TAG suffix)
    wandb_project: str | None = "qwen35-35b"
    wandb_name: str | None = None
    run_tag: str = "35b-mixed-filtered-e30-lr1e-7-lora32"
    recipe_name: str = "recipe_sl_basic"

    log_dir: str
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps: int | None = None


def cli_main(cli: CLIConfig):
    model_slug = cli.model_name.replace("/", "-")
    date = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M")
    run_name = f"{cli.run_tag}-{model_slug}-{date}"

    log_path = str(Path(cli.log_dir) / run_name)
    wandb_name = cli.wandb_name if cli.wandb_name is not None else cli.run_tag

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cli.model_name,
        explicit_renderer_name=cli.renderer_name,
        load_checkpoint_path=cli.load_checkpoint_path,
        base_url=cli.base_url,
    )

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        max_length=cli.max_length,
        batch_size=cli.batch_size,
        train_on_what=cli.train_on_what,
    )
    dataset_builder = FromParquetFileBuilder(
        common_config=common,
        file_path=cli.dataset_path,
        test_size=cli.test_size,
        shuffle_seed=cli.shuffle_seed,
    )

    config = train.Config(
        log_path=log_path,
        model_name=cli.model_name,
        recipe_name=cli.recipe_name,
        renderer_name=renderer_name,
        load_checkpoint_path=cli.load_checkpoint_path,
        dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate,
        lr_schedule=cli.lr_schedule,
        num_epochs=cli.num_epochs,
        lora_rank=cli.lora_rank,
        save_every=cli.save_every,
        eval_every=cli.eval_every,
        infrequent_eval_every=cli.infrequent_eval_every,
        wandb_project=cli.wandb_project,
        wandb_name=wandb_name,
        base_url=cli.base_url,
        max_steps=cli.max_steps,
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    cli_main(chz.entrypoint(CLIConfig))
