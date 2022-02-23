from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_config, setup_imports
from ocpmodels.datasets import data_list_collater

setup_imports()

config_file="eval_config_schnet.yml"
checkpoint_file="checkpoints/schnet_10k.pt"

config = load_config(config_file)[0]
config["identifier"]=""
config["local_rank"]=0

trainer = registry.get_trainer_class(
    config.get("trainer", "energy")
)(
    task=config["task"],
    model=config["model"],
    dataset=config["dataset"],
    optimizer=config["optim"],
    identifier=config["identifier"],
    timestamp_id=config.get("timestamp_id", None),
    run_dir=config.get("run_dir", "./"),
    is_debug=config.get("is_debug", False),
    is_vis=config.get("is_vis", False),
    print_every=config.get("print_every", 10),
    seed=config.get("seed", 0),
    logger=config.get("logger", "tensorboard"),
    local_rank=config["local_rank"],
    amp=config.get("amp", False),
    cpu=config.get("cpu", False),
    slurm=config.get("slurm", {}),
)
trainer.load_checkpoint(checkpoint_file)


def setup(config_file, checkpoint_file):
    config = load_config(config_file)[0]
    config["identifier"]=""
    config["local_rank"]=0

    trainer = registry.get_trainer_class(
        config.get("trainer", "energy")
    )(
        task=config["task"],
        model=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        timestamp_id=config.get("timestamp_id", None),
        run_dir=config.get("run_dir", "./"),
        is_debug=config.get("is_debug", False),
        is_vis=config.get("is_vis", False),
        print_every=config.get("print_every", 10),
        seed=config.get("seed", 0),
        logger=config.get("logger", "tensorboard"),
        local_rank=config["local_rank"],
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
        slurm=config.get("slurm", {}),
    )
    trainer.load_checkpoint(checkpoint_file)
    return config, trainer