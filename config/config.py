import yaml
from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfiguration:
    name: str
    hidden_layers: List[int]
    activation_function: str
    epoch_number: int
    learning_rate: float


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path

    def load_configurations(self) -> List[ModelConfiguration]:
        with open(self.config_path, "r") as file:
            config_data = yaml.safe_load(file)

        model_configs = [
            ModelConfiguration(
                name=model["name"],
                hidden_layers=model["hidden_layers"],
                activation_function=model["activation_function"],
                epoch_number=model["epoch_number"],
                learning_rate=model["learning_rate"]
            )
            for model in config_data["models"]
        ]

        return model_configs


# 使用方法
if __name__ == "__main__":
    config_loader = ConfigLoader("config/model-configuration.yaml")
    model_configs = config_loader.load_configurations()

    for config in model_configs:
        print(config)
