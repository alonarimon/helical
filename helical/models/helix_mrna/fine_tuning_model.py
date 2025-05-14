import os
from typing import Literal, Optional
from helical.models.base_models import (
    HelicalBaseFineTuningHead,
    HelicalBaseFineTuningModel,
)
from helical.models.helix_mrna.model import HelixmRNA, HelixmRNAConfig
from datasets import Dataset
from transformers import get_scheduler
import torch
from torch import optim
from torch.nn.modules import loss
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import wandb
import logging

LOGGER = logging.getLogger(__name__)


class HelixmRNAFineTuningModel(HelicalBaseFineTuningModel, HelixmRNA):
    """HelixmRNAFineTuningModel
    Fine-tuning model for the Helix-mRNA model. This model can be used to fine-tune the Helix-mRNA model on a downstream task.

    Example
    ----------
    ```python
    from helical.models.helix_mrna import HelixmRNAFineTuningModel, HelixmRNAConfig
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_sequences = ["EACU"*20, "EAUG"*20, "EUGC"*20, "ECUG"*20, "EUUG"*20]
    labels = [0, 2, 2, 0, 1]

    helix_mrna_config = HelixmRNAConfig(batch_size=5, device=device, max_length=100)
    helix_mrna_fine_tune = HelixmRNAFineTuningModel(helix_mrna_config=helix_mrna_config, fine_tuning_head="classification", output_size=3)

    train_dataset = helix_mrna_fine_tune.process_data(input_sequences)

    helix_mrna_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

    outputs = helix_mrna_fine_tune.get_outputs(train_dataset)
    print(outputs.shape)
    ```

    Parameters
    ----------
    helix_mrna_config : HelixmRNAConfig
        The configuration object for the Helix-mRNA model. The same config object can be used for both the HelixmRNA and HelixmRNAFineTuningModel.
    fine_tuning_head : Literal["classification", "regression"] | HelicalBaseFineTuningHead
        The type of fine-tuning head to use for the model. This can be either a classification or regression head, or a custom fine-tuning head.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the fine_tuning_head is a string specified task. For a classification task this is number of unique classes.

    Methods
    ----------
    train(train_dataset, train_labels, optimizer, optimizer_params, loss_function, epochs, freeze_layers, validation_dataset, validation_labels, lr_scheduler_params)
        Fine-tunes the Helix-mRNA model on the given dataset.
    get_outputs(dataset)
        Returns the outputs of the model for the given dataset.
    """

    def __init__(
        self,
        helix_mrna_config: HelixmRNAConfig,
        fine_tuning_head: (
            Literal["classification", "regression"] | HelicalBaseFineTuningHead
        ),
        output_size: Optional[int] = None,
    ):
        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        HelixmRNA.__init__(self, helix_mrna_config)

        self.fine_tuning_head.set_dim_size(self.pretrained_config.hidden_size * 2)

    def _forward(self, input_ids, special_tokens_mask=None):

        transformer_outputs = self.model(
            input_ids=input_ids, attention_mask=1 - special_tokens_mask
        )

        hidden_states = transformer_outputs[0]

        batch_size = input_ids.shape[0]

        ## We Average the hidden states to get the pooled output
        sequence_lengths = (
            torch.eq(input_ids, self.pretrained_config.pad_token_id).int().argmax(-1)
            - 1
        )
        sequence_beginnings = (
            (~torch.eq(input_ids, self.pretrained_config.pad_token_id))
            .int()
            .argmax(-1)
            .to(hidden_states.device)
        )
        sequence_lengths = sequence_lengths % input_ids.shape[-1] - 1
        sequence_lengths = sequence_lengths.to(hidden_states.device)
        # print(sequence_lengths)
        # print(input_ids)
        mask = (
            sequence_beginnings[:, None]
            < torch.arange(hidden_states.size(1), device=hidden_states.device)[None, :]
        )  # < sequence_lengths[:, None]
        masked_tensor = hidden_states * mask.unsqueeze(-1)
        sum_tensor = masked_tensor.sum(dim=1)
        mean_states = sum_tensor / (
            sequence_lengths.unsqueeze(-1).float()
            - sequence_beginnings.unsqueeze(-1).float()
        )

        selected_last_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device), sequence_lengths
        ]

        hidden_states = torch.cat([selected_last_hidden_states, mean_states], dim=-1)

        logits = self.fine_tuning_head(hidden_states)

        return logits
    
    def optimize_conservatism_embeddings(
    self,
    embeds_input: torch.Tensor,
    input_ids: torch.Tensor,
    model: nn.Module,
    special_tokens_mask: torch.Tensor,
    steps: int = 50,
    lr: float = 0.05,
    entropy_coeff: float =  0.9,
    ) -> torch.Tensor:
        x_opt = embeds_input.clone().detach().requires_grad_(True)
        for _ in range(steps):
            with torch.no_grad():
                transformer_out = self.model(inputs_embeds=x_opt, attention_mask=1 - special_tokens_mask)[0]

            transformer_out.requires_grad_(True)

            # Faster pooling: simple mean pooling for speed
            pooled = transformer_out.mean(dim=1)

            # Directly score pooled representations
            score = self.fine_tuning_head(pooled).mean()

            grad = torch.autograd.grad(score, transformer_out, retain_graph=False)[0]

            # average gradient over sequence length for speed
            with torch.no_grad():
                grad_mean = grad.mean(dim=1, keepdim=True)
                x_opt += lr * grad_mean

        return x_opt.detach()
    
    def embed_forward(self, x, input_ids, special_tokens_mask):
        with torch.no_grad():
            transformer_out = self.model(inputs_embeds=x, attention_mask=1 - special_tokens_mask)[0]

        batch_size = transformer_out.shape[0]
        sequence_lengths = (
            torch.eq(input_ids, self.pretrained_config.pad_token_id).int().argmax(-1) - 1
        )
        sequence_beginnings = (
            (~torch.eq(input_ids, self.pretrained_config.pad_token_id)).int().argmax(-1).to(transformer_out.device)
        )
        sequence_lengths = sequence_lengths % input_ids.shape[-1] - 1
        sequence_lengths = sequence_lengths.to(transformer_out.device)

        mask = (
            sequence_beginnings[:, None]
            < torch.arange(transformer_out.size(1), device=transformer_out.device)[None, :]
        )
        masked_tensor = transformer_out * mask.unsqueeze(-1)
        sum_tensor = masked_tensor.sum(dim=1)
        mean_states = sum_tensor / (
            sequence_lengths.unsqueeze(-1).float()
            - sequence_beginnings.unsqueeze(-1).float()
        )

        selected_last_hidden_states = transformer_out[
            torch.arange(batch_size, device=transformer_out.device), sequence_lengths
        ]

        pooled = torch.cat([selected_last_hidden_states, mean_states], dim=-1)

        return pooled  # Now has shape (batch_size, 512), correct for your fine-tuning head



    def train_fine_tune(
        self,
        train_dataset: Dataset,
        train_labels: np.ndarray,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {"lr": 0.0001},
        loss_function: loss = loss.CrossEntropyLoss(),
        epochs: int = 1,
        trainable_layers: int = 2,
        validation_dataset: Optional[Dataset] = None,
        validation_labels: Optional[np.ndarray] = None,
        lr_scheduler_params: Optional[dict] = None,
        return_loss: bool = False,
        save_dir: Optional[str] = None,
        use_com_loss=False,
        com_steps: int = 50,
        com_lr: float = 0.05,
        com_entropy_coeff: float = 0.9,
        com_overestimation_limit=2.0,
        com_aplha_init=0.1,
        com_alpha_lr=0.01,

    ):
        """Fine-tunes the Helix-mRNA model on the given dataset.

        Parameters
        ----------
        train_dataset : Dataset
            A helical processed dataset for fine-tuning
        train_labels : np.ndarray
            The labels for the training dataset
        optimizer : torch.optim, default=torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict, optional, default={'lr': 0.0001}
            The optimizer parameters to be used for the optimizer specified. This list should NOT include model parameters.
            e.g. optimizer_params = {'lr': 0.0001}
        loss_function : torch.nn.modules.loss, default=torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        epochs : int, optional, default=10
            The number of epochs to train the model
        trainable_layers : int, optional, default=2
            The number of layers to train in the model. The last n layers will be trained and the rest will be frozen.
        validation_dataset : Dataset, default=None
            A helical processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        validation_labels : np.ndarray, default=None
            The labels for the validation dataset. This is required if a validation dataset is specified.
        lr_scheduler_params : dict, default=None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0 }. num_steps will be calculated based on the number of epochs and the length of the training dataset.

        """
        LOGGER.info(f"Fine-Tuning Helix-mRNA Model on {self.config['device']}")

        # initialise optimizer
        optimizer = optimizer(self.parameters(), **optimizer_params)

        # set labels for the dataset
        train_dataset = self._add_data_column(
            train_dataset, "labels", np.array(train_labels)
        )
        if validation_labels is not None and validation_dataset is not None:
            validation_dataset = self._add_data_column(
                validation_dataset, "labels", np.array(validation_labels)
            )

        if trainable_layers > 0:
            LOGGER.info(
                f"Unfreezing the last {trainable_layers} layers of the Helix_mRNA model."
            )

            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.layers[-trainable_layers:].parameters():
                param.requires_grad = True

        self.to(self.config["device"])

        self.model.train()
        self.fine_tuning_head.train()

        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
        )

        lr_scheduler = None
        if lr_scheduler_params is not None:
            lr_scheduler = get_scheduler(
                optimizer=optimizer,
                num_training_steps=epochs * len(train_dataloader),
                **lr_scheduler_params,
            )

        if validation_dataset is not None:
            validation_dataloader = DataLoader(
                validation_dataset,
                collate_fn=self._collate_fn,
                batch_size=self.config["val_batch_size"],
            )

        LOGGER.info("Starting Fine-Tuning")
        epoch_losses_train = []
        epoch_losses_validation = []

        for j in range(epochs):
            training_loop = tqdm(train_dataloader, desc="Fine-Tuning")
            batch_loss = 0.0
            batches_processed = 0

            for batch in training_loop:
                
                input_ids = batch["input_ids"].to(self.config["device"])
                special_tokens_mask = batch["special_tokens_mask"].to(
                    self.config["device"]
                )
                labels = batch["labels"].to(self.config["device"])
                labels = labels.unsqueeze(-1)
                outputs = self._forward(
                    input_ids=input_ids, special_tokens_mask=special_tokens_mask
                )


                if not use_com_loss:
                    print(f"inputs: {input_ids[:1]}")
                    print(f"outputs: {outputs[:1]}")
                    print(f"labels: {labels[:1]}")
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    
                else:

                    # === Extract embeddings from inputs ===
                    inputs_embeds = self.model.embeddings(input_ids)

                    # === Find negative samples (adversarial embeddings) ===
                    perturbed_embeds = self.optimize_conservatism_embeddings(
                        embeds_input=inputs_embeds,
                        model=self.fine_tuning_head,
                        input_ids=input_ids,
                        steps=com_steps,
                        lr=com_lr,
                        entropy_coeff=com_entropy_coeff,
                        special_tokens_mask=special_tokens_mask,
                    )

                    # === Predictions on original (positive) and perturbed (negative) embeddings ===
                    pred_pos = outputs
                    pooled_adv = self.embed_forward(
                        x=perturbed_embeds,
                        input_ids=input_ids,
                        special_tokens_mask=special_tokens_mask
                    )
                    pred_neg = self.fine_tuning_head(pooled_adv)

                    # === Calculate overestimation ===
                    overestimation = (pred_neg - pred_pos).detach()

                    # === Initialize alpha (add these lines outside the training loop, at the top of train_fine_tune) ===
                    if not hasattr(self, 'log_alpha'):
                        self.log_alpha = torch.tensor(np.log(com_aplha_init), dtype=torch.float32, requires_grad=True, device=self.config["device"])
                        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=com_alpha_lr)

                    # === Compute total losses ===
                    alpha = self.log_alpha.exp()

                    base_loss = loss_function(pred_pos, labels)
                    model_loss = base_loss + (alpha * overestimation).mean()
                    alpha_loss = (alpha * com_overestimation_limit - alpha * overestimation).mean()
                    loss = model_loss

                    # === Logging additional COM metrics (optional but recommended) ===
                    wandb.log({
                        "train/overestimation": overestimation.mean().item(),
                        "train/alpha": alpha.item(),
                    })

                    # === Gradient updates for model ===
                    optimizer.zero_grad()
                    model_loss.backward(retain_graph=True)  # retain graph for alpha optimization
                    optimizer.step()

                    # === Gradient updates for alpha ===
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                
                batch_loss += loss.item()
                batches_processed += 1.0
                
            training_loop.set_postfix({"train_loss": batch_loss / batches_processed})
            epoch_losses_train.append(batch_loss / batches_processed)
            wandb.log({"epoch": j, "train_loss": batch_loss / batches_processed})
            del training_loop

            if validation_dataset is not None:
                testing_loop = tqdm(
                    validation_dataloader, desc="Fine-Tuning Validation"
                )
                val_loss = 0.0
                count = 0.0
                for test_batch in testing_loop:
                    input_ids = test_batch["input_ids"].to(self.config["device"])
                    special_tokens_mask = test_batch["special_tokens_mask"].to(
                        self.config["device"]
                    )
                    labels = test_batch["labels"].to(self.config["device"])
                    labels = labels.unsqueeze(-1)

                    with torch.no_grad():
                        outputs = self._forward(
                            input_ids=input_ids, special_tokens_mask=special_tokens_mask
                        )

                    val_loss += loss_function(outputs, labels).item()
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss / count})
                    wandb.log({"epoch": j, "val_loss": val_loss / count})

                    del test_batch
                    del outputs

                epoch_losses_validation.append(val_loss / count)
                del testing_loop

            if save_dir is not None and j % 5 == 0:
                model_path = f"{save_dir}/model_epoch_{j+1}"
                self.save_model(model_path)

        LOGGER.info(f"Fine-Tuning Complete. Epochs: {epochs}")
        if return_loss:
            return epoch_losses_train, epoch_losses_validation

    def get_outputs(self, dataset: Dataset, verbose = False) -> np.ndarray:
        """
        Returns the outputs of the model for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset object returned by the `process_data` function.

        Returns
        ----------
        np.ndarray
            The outputs of the model for the given dataset
        """
        dataloader = DataLoader(
            dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
            shuffle=False,
        )
        outputs = []

        self.model.to(self.config["device"])

        progress_bar = tqdm(dataloader, desc="Generating outputs", disable=not verbose)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.config["device"])
            special_tokens_mask = batch["special_tokens_mask"].to(self.config["device"])

            with torch.no_grad():
                output = self._forward(
                    input_ids, special_tokens_mask=special_tokens_mask
                )

            outputs.append(output.cpu().numpy())

            del batch
            del output

        return np.concatenate(outputs)

    def _add_data_column(self, dataset, column_name, data):
        if len(data.shape) > 1:
            for i in range(len(data[0])):  # Assume all inner lists are the same length
                dataset = dataset.add_column(f"{column_name}", [row[i] for row in data])
        else:  # If 1D
            dataset = dataset.add_column(column_name, data)
        return dataset

    def save_model(self, save_dir: str):
        """Saves the model to the specified directory.

        Parameters
        ----------
        save_dir : str
            The directory to save the model to.
        """
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "base_model.pt"))
        torch.save(self.fine_tuning_head.state_dict(), os.path.join(save_dir, "head.pt"))
        torch.save(self.config, os.path.join(save_dir, "config.pt"))
        LOGGER.info(f"Model saved to {save_dir}")

    def load_model(self, load_dir: str):
        """Loads the model from the specified directory.

        Parameters
        ----------
        load_dir : str
            The directory to load the model from.
        """
        self.model.load_state_dict(torch.load(os.path.join(load_dir, "base_model.pt")))
        self.fine_tuning_head.load_state_dict(
            torch.load(os.path.join(load_dir, "head.pt"))
        )
        # config_dict = torch.load(os.path.join(load_dir, "config.pt"))
        # self.config = HelixmRNAConfig(batch_size=config_dict["batch_size"],
        #                               device=self.device,
        #                               max_length=config_dict["input_size"],
        #                               val_batch_size=config_dict["val_batch_size"],
        #                                nproc=config_dict["nproc"]) # TODO: need this?
        LOGGER.info(f"Model loaded from {load_dir}")
