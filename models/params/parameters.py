from dataclasses import dataclass


"""
Holds data about a model information
"""


@dataclass
class ModelInfo:
    scale: int
    features: int
    blockCount: int
    lr: float
    epoch: int


"""
Constructs a model path given data
"""


def ConstructPath(
    scale: int, features: int, blockCount: int, lr: float, epoch: int
) -> str:
    return f"models/params/{scale}_{features}_{blockCount}_{lr}_{epoch}.pt"


"""
Parses a model path back into info about the model
"""


def ParsePath(name: str) -> ModelInfo:
    name = name[:-3]
    name = name.split("/")[-1]
    name = name.split("_")
    scale = int(name[0])
    features = int(name[1])
    blockCount = int(name[2])
    lr = float(name[3])
    epoch = int(name[4])
    return ModelInfo(scale, features, blockCount, lr, epoch)


def main():
    name = ConstructPath(4, 64, 12, 1e-4, 20)
    print(name)
    print(ParsePath(name))


if __name__ == "__main__":
    main()
