from helical.models.caduceus import CaduceusFineTuningModel, CaduceusConfig
import hydra
from omegaconf import DictConfig
import wandb
from torch import nn as torch_loss

@hydra.main(
    version_base=None,
    config_path="../run_models/configs",
    config_name="caduceus_config",
)
def run_fine_tuning(cfg: DictConfig):
    input_sequences = ["ACT" * 20, "ATG" * 10, "ATG" * 20, "CTG" * 10, "TTG" * 20]
    labels = [0.0, 1.0, 0.5, 0.75, 0.25]

    caduceus_config = CaduceusConfig(nproc=0)
    caduceus_fine_tune = CaduceusFineTuningModel(
        caduceus_config=caduceus_config,
        fine_tuning_head="regression",
        output_size=1,
    )

    train_dataset = caduceus_fine_tune.process_data(input_sequences)

    caduceus_fine_tune.train(train_dataset=train_dataset, train_labels=labels, loss_function=torch_loss.MSELoss())

    outputs = caduceus_fine_tune.get_outputs(train_dataset)
    print(outputs.shape)
    print(outputs)


if __name__ == "__main__":
    wandb.init(
    project="bioseq_qd_design",
    name="caduceus_fine_tuning",
    mode="disabled",
    )
    run_fine_tuning()
