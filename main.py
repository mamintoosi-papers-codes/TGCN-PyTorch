import argparse
import yaml
import logging
import torch
import os

import models  # noqa: F401 just to ensure we import GCN, GRU, TGCN
from tasks.supervised import SupervisedForecastTask
from utils.data.spatiotemporal_csv_data import SpatioTemporalCSVData
from utils.logging import format_logger, output_logger_to_file


def train_and_validate(
    model_task: SupervisedForecastTask,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    num_epochs=50,
    batch_size=32,
    device="cuda",
    logger=logging.getLogger(__name__),
):
    """
    Manual training + validation loop (pure PyTorch).
    """
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    model_task.model.to(device)
    if model_task.regressor is not None:
        model_task.regressor.to(device)

    optimizer = model_task.configure_optimizer()

    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model_task.model.train()
        if model_task.regressor is not None:
            model_task.regressor.train()

        total_train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model_task.training_step((x, y))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ---- Validation ----
        model_task.model.eval()
        if model_task.regressor is not None:
            model_task.regressor.eval()

        val_metrics = model_task.validation_epoch(val_loader, device)
        logger.info(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.6f}, "
            f"Val Loss: {val_metrics['val_loss']:.6f}, "
            f"RMSE: {val_metrics['RMSE']:.6f}, "
            f"MAE: {val_metrics['MAE']:.6f}, "
            f"Accuracy: {val_metrics['accuracy']:.4f}, "
            f"R2: {val_metrics['R2']:.4f}, "
            f"Expl.Var: {val_metrics['ExplainedVar']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gcn.yaml", help="Path to config YAML.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--device", type=str, default="cuda", help="Which device to use (e.g. cuda or cpu)")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    format_logger(logger)
    if args.log_path:
        output_logger_to_file(logger, args.log_path)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {args.config}: {config}")

    num_epochs = config["fit"]["trainer"]["max_epochs"]
    batch_size = config["fit"]["data"]["batch_size"]
    seq_len = config["fit"]["data"]["seq_len"]
    pre_len = config["fit"]["data"]["pre_len"]
    learning_rate = config["fit"]["model"]["learning_rate"]
    weight_decay = config["fit"]["model"]["weight_decay"]
    loss_name = config["fit"]["model"]["loss"]

    model_class_path = config["fit"]["model"]["model"]["class_path"]  # e.g. "models.GCN"
    model_init_args = config["fit"]["model"]["model"].get("init_args", {})

    dataset_name = config["fit"]["data"]["dataset_name"]
    data_module = SpatioTemporalCSVData(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pre_len=pre_len,
        split_ratio=0.8,
        normalize=True,
    )
    train_dataset, val_dataset = data_module.get_datasets()

    # Force required arguments depending on the model
    model_cls_name = model_class_path.split(".")[-1]  # e.g. "GCN"
    ModelClass = getattr(models, model_cls_name)

    if model_cls_name in ("GCN", "TGCN"):
        model_init_args["adj"] = data_module.adj
        model_init_args["seq_len"] = seq_len
    elif model_cls_name == "GRU":
        model_init_args["num_nodes"] = data_module.num_nodes

    model = ModelClass(**model_init_args)

    model_task = SupervisedForecastTask(
        model=model,
        loss=loss_name,
        pre_len=pre_len,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        feat_max_val=data_module.feat_max_val,
    )

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    logger.info(f"Using device: {device}")

    train_and_validate(
        model_task=model_task,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        logger=logger,
    )

    logger.info("Finished training!")

if __name__ == "__main__":
    main()
