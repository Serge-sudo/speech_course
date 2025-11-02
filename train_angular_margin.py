#!/usr/bin/env python3
from pathlib import Path
from speaker_recognition import ModuleParams, ModelParams, main

params = ModuleParams(
    dataset_dir=Path("./data/train"),
    validation_dir=Path("./data/dev"),

    checkpoints_dir=Path("./data/checkpoints/angular_margin"),
    log_dir=Path("./data/logs/angular_margin"),

    model_params=ModelParams(),

    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    n_epochs=80,
    batch_size=16,
    num_workers=4,
    learning_rate=3e-4,
    loss_function="angular_margin",
    angular_margin=0.3,
    angular_scale=20.0,
    validation_frequency=5,
    use_cache=True,
)

if __name__ == "__main__":
    main(params)
    print("Training complete!")
