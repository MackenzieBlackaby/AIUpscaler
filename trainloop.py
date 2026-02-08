import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.py.SuperResolution import SupResNet
from dataset.ImageSet import ImageSet, ImagePair
from dataset.DataDownloader import DownloadData
from models.params.paths import ConstructPath


def extractPairs(batch: list[ImagePair]):
    lr = torch.stack([pair.lowRes for pair in batch])
    hr = torch.stack([pair.highRes for pair in batch])
    return lr, hr


def main():
    trainImageDir = DownloadData()
    scale = 4
    hrCrop = 192
    blockCount = 12
    features = 64
    batchSize = 16
    epochs = 20
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = ImageSet(trainImageDir, scale, hrCrop)

    validationPercentage = 0.05
    validationLength = max(1, int(len(dataset) * validationPercentage))
    trainLength = len(dataset) - validationLength
    trainSet, validationSet = random_split(dataset, [trainLength, validationLength])

    trainLoader = DataLoader(
        trainSet,
        batch_size=batchSize,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=extractPairs,
    )
    validationLoader = DataLoader(
        validationSet,
        batch_size=batchSize,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
        drop_last=False,
        collate_fn=extractPairs,
    )

    model = SupResNet(scale=scale, blockCount=blockCount, features=features).to(device)

    lossfn = nn.L1Loss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        runningLoss = 0.0
        for step, (lrImgs, hrImgs) in enumerate(trainLoader, start=1):
            lrImgs = lrImgs.to(device, non_blocking=True)
            hrImgs = hrImgs.to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            srImgs = model(lrImgs)
            loss = lossfn(srImgs, hrImgs)
            loss.backward()
            optimiser.step()

            runningLoss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch} | step {step} | train L1: {runningLoss/100:.5f}")
                runningLoss = 0.0

        model.eval()
        val_loss = 0.0
        batches = 0

        with torch.no_grad():
            for lrImgs, hrImgs in validationLoader:
                lrImgs = lrImgs.to(device, non_blocking=True)
                hrImgs = hrImgs.to(device, non_blocking=True)

                sr_imgs = model(lrImgs)
                loss = lossfn(sr_imgs, hrImgs)

                val_loss += loss.item()
                batches += 1

        print(f"Epoch {epoch} | val L1: {val_loss/max(1,batches):.5f}")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimiser.state_dict(),
            },
            f=ConstructPath(scale, features, blockCount, lr, epoch),
        )


if __name__ == "__main__":
    main()
