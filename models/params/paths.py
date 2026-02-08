from dataclasses import dataclass

@dataclass
class ModelInfo:
    """
    Holds data about a model configuration

    :param scale: The upscaling factor
    :type scale: int
    :param features: The feature count of the model (micro complexity)
    :type features: int
    :param blockCount: The block count of the model (macro complexity)
    :type blockCount: int
    :param lr: The learning rate of the model
    :type lr: float
    :param epoch: The epoch this model was trained until
    :type epoch: int
    """

    scale: int
    features: int
    blockCount: int
    lr: float
    epoch: int


def ConstructPath(
    scale: int, features: int, blockCount: int, lr: float, epoch: int
) -> str:
    """
    Constructs a path, given data about a model configuration, to its specific checkpoint file relative to the current folder

    :param scale: The upscaling factor
    :type scale: int
    :param features: The feature count of the model (micro complexity)
    :type features: int
    :param blockCount: The block count of the model (macro complexity)
    :type blockCount: int
    :param lr: The learning rate of the model
    :type lr: float
    :param epoch: The epoch this model was trained until
    :type epoch: int
    :return: A constructed relative path to the model checkpoint file
    :rtype: str
    """
    return f"models/params/{scale}_{features}_{blockCount}_{lr}_{epoch}.pt"


def ConstructPath(scale: int, features: int, blockCount: int, lr: float) -> str:
    """
    Constructs a path, given data about a model configuration, to its specific final checkpoint file relative to the current folder

    :param scale: The upscaling factor
    :type scale: int
    :param features: The feature count of the model (micro complexity)
    :type features: int
    :param blockCount: The block count of the model (macro complexity)
    :type blockCount: int
    :param lr: The learning rate of the model
    :type lr: float
    :return: A constructed relative path to the model checkpoint file
    :rtype: str
    """
    return f"models/params/{scale}_{features}_{blockCount}_{lr}.pt"


def ParsePath(name: str) -> ModelInfo:
    """
    Parses a model path/filename to get specifics about the configuration

    :param name: Description
    :type name: str
    :return: Description
    :rtype: ModelInfo
    """
    name = name[:-3]
    name = name.split("/")[-1]
    name = name.split("_")
    scale = int(name[0])
    features = int(name[1])
    blockCount = int(name[2])
    lr = float(name[3])
    epoch = int(name[4]) if len(name) == 5 else -1
    return ModelInfo(scale, features, blockCount, lr, epoch)


def main():
    """
    Main function for debugging this module
    """
    name = ConstructPath(4, 64, 12, 1e-4, 20)
    print(name)
    print(ParsePath(name))


if __name__ == "__main__":
    main()
