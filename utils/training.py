import time

import torch

from utils.config import device


# =========================================================
# Batch Utilities
# =========================================================

def move_batch_to_device(
    batch,
    device=device
):
    """
    Move a batch to the target device.

    Supports:

    1. TensorDataset style
       (inputs, labels)

    2. Hugging Face style
       {
           "input_ids": ...,
           "attention_mask": ...,
           "labels": ...
       }
    """

    # -----------------------------------------------------
    # Hugging Face / dictionary batch
    # -----------------------------------------------------

    if isinstance(batch, dict):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        labels = batch.pop(
            "labels"
        )

        return batch, labels

    # -----------------------------------------------------
    # TensorDataset style
    # -----------------------------------------------------

    inputs, labels = batch

    inputs = inputs.to(device)
    labels = labels.to(device)

    return inputs, labels


# =========================================================
# Forward Pass
# =========================================================

def forward_batch(
    model,
    batch,
    device=device
):
    """
    Execute a forward pass for both
    TensorDataset and Hugging Face style batches.
    """

    inputs, labels = move_batch_to_device(
        batch,
        device
    )

    if isinstance(inputs, dict):

        outputs = model(
            **inputs
        )

    else:

        outputs = model(
            inputs
        )

    return outputs, labels


# =========================================================
# Train One Epoch
# =========================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device=device
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:

        optimizer.zero_grad()

        outputs, labels = forward_batch(
            model,
            batch,
            device
        )

        # -------------------------------------------------
        # Hugging Face model output
        # -------------------------------------------------

        if hasattr(
            outputs,
            "logits"
        ):

            logits = outputs.logits

        else:

            logits = outputs

        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        loss = criterion(
            logits,
            labels
        )

        # -------------------------------------------------
        # Backpropagation
        # -------------------------------------------------

        loss.backward()

        optimizer.step()

        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    average_loss = (
        total_loss
        / total_samples
    )

    return average_loss


# =========================================================
# Evaluation
# =========================================================

def evaluate(
    model,
    dataloader,
    criterion,
    device=device
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch in dataloader:

            outputs, labels = forward_batch(
                model,
                batch,
                device
            )

            # -------------------------------------------------
            # Get logits
            # -------------------------------------------------

            if hasattr(
                outputs,
                "logits"
            ):

                logits = outputs.logits

            else:

                logits = outputs

            # -------------------------------------------------
            # Loss
            # -------------------------------------------------

            loss = criterion(
                logits,
                labels
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = (
        total_loss
        / total_samples
    )

    return (
        average_loss,
        all_labels,
        all_predictions
    )


# =========================================================
# Training
# =========================================================

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    device=device
):

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": []
    }

    best_macro_f1 = -1.0
    best_state_dict = None

    start_time = time.perf_counter()

    for epoch in range(
        1,
        epochs + 1
    ):

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        (
            val_loss,
            y_true,
            y_pred
        ) = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        # -------------------------------------------------
        # Macro F1
        # -------------------------------------------------

        from utils.metrics import calculate_metrics

        metrics = calculate_metrics(
            y_true,
            y_pred
        )

        val_macro_f1 = (
            metrics["macro_f1"]
        )

        # -------------------------------------------------
        # History
        # -------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_macro_f1"].append(
            val_macro_f1
        )

        # -------------------------------------------------
        # Best Model
        # -------------------------------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = (
                val_macro_f1
            )

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        # -------------------------------------------------
        # Print
        # -------------------------------------------------

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Macro F1: {val_macro_f1:.4f}"
        )

    training_time = (
        time.perf_counter()
        - start_time
    )

    # -----------------------------------------------------
    # Restore Best Model
    # -----------------------------------------------------

    if best_state_dict is not None:

        model.load_state_dict(
            best_state_dict
        )

    return (
        model,
        history,
        best_macro_f1,
        training_time
    )