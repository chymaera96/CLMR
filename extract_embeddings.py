import os
import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose, RandomResizedCrop
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from clmr.datasets import get_dataset
from clmr.data import ContrastiveDataset
from clmr.evaluation import evaluate
from clmr.models import SampleCNN
from clmr.modules import ContrastiveLearning, LinearEvaluation
from clmr.utils import (
    yaml_config_hook,
    load_encoder_checkpoint,
    load_finetuner_checkpoint,
)

parser = argparse.ArgumentParser(description="SimCLR")
parser = Trainer.add_argparse_args(parser)

config = yaml_config_hook("./config/config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))

parser.add_argument("--data_dir", default='../DLAM_coursework/data/fma_downstream', type=str)
parser.add_argument("--ckp", default='clmr_magnatagatune_mlp/clmr_epoch=10000.ckpt', type=str)
parser.add_argument("--out_dir", default='../DLAM_coursework/features', type=str)


def extract_representations(encoder, dataloader):

    representations = []
    ys = []
    for x, y in tqdm(dataloader):
        with torch.no_grad():
            h0 = encoder(x)
            representations.append(h0)
            ys.append(y)

    if len(representations) > 1:
        representations = torch.cat(representations, dim=0)
        ys = torch.cat(ys, dim=0)
    else:
        representations = representations[0]
        ys = ys[0]

        return ys


def main():
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    args.accelerator = None

    if not os.path.exists(args.ckp):
        raise FileNotFoundError("That checkpoint does not exist")
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    print("Preprocessing audio and creating dataset object ...")
    audio_dataset = get_dataset(dataset_dir=args.data_dir, subset="train")
    for idx in range(len(audio_dataset)):
        audio_dataset.preprocess(idx, args.sample_rate)

    contrastive_dataset = ContrastiveDataset(
        audio_dataset,
        input_shape=(1, args.audio_length),
        transform=None,

    )

    audio_loader = DataLoader(
        contrastive_dataset,
        batch_size=1,
        num_workers=args.workers,
        shuffle=False,
    )

    print("Creating SampleCNN encoder ...")
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
        out_dim=audio_dataset.n_classes,
    )

    n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer

    print(f"Loading checkpoint from {args.ckp}")
    state_dict = load_encoder_checkpoint(args.ckp, audio_dataset.n_classes)

    encoder.load_state_dict(state_dict)

    cl = ContrastiveLearning(args, encoder)
    cl.eval()
    cl.freeze()

    if not os.exists(args.out_dir):
        os.mkdir(args.out_dir)

    print(f"Extracting embeddings ...")
    embs = extract_representations(cl.encoder, audio_loader)
    fname = args.data_dir.split('/')[-1]

    out_path = os.path.join(args.out_dir, f'CLMR_features_{fname}.pt')
    print(f"Saving embeddings at {out_path}")
    torch.save(embs, out_path)


if __name__ == '__main__':
    main()