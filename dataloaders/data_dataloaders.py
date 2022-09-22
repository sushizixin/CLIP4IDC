import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_clevr_caption import CLEVR_DataLoader as CLEVR_Caption_DataLoader
from dataloaders.dataloader_spot_caption import SPOT_DataLoader as SPOT_Caption_DataLoader
from dataloaders.dataloader_clevr_retrieval import CLEVR_DataLoader
from dataloaders.dataloader_spot_retrieval import SPOT_DataLoader


def dataloader_clevr_train(args, tokenizer):
    if args.task_type == "retrieval":
        DataSet_DataLoader = CLEVR_DataLoader
    else:
        DataSet_DataLoader = CLEVR_Caption_DataLoader

    clevr_dataset = DataSet_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(clevr_dataset)
    dataloader = DataLoader(
        clevr_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(clevr_dataset), train_sampler

def dataloader_clevr_test(args, tokenizer, subset="test"):
    if args.task_type == "retrieval":
        DataSet_DataLoader = CLEVR_DataLoader
    else:
        DataSet_DataLoader = CLEVR_Caption_DataLoader

    clevr_testset = DataSet_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )
    dataloader_clevr = DataLoader(
        clevr_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_clevr, len(clevr_testset)

def dataloader_spot_train(args, tokenizer):
    if args.task_type == "retrieval":
        DataSet_DataLoader = SPOT_DataLoader
    else:
        DataSet_DataLoader = SPOT_Caption_DataLoader

    spot_dataset = DataSet_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(spot_dataset)
    dataloader = DataLoader(
        spot_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(spot_dataset), train_sampler

def dataloader_spot_test(args, tokenizer, subset="test"):
    if args.task_type == "retrieval":
        DataSet_DataLoader = SPOT_DataLoader
    else:
        DataSet_DataLoader = SPOT_Caption_DataLoader

    spot_testset = DataSet_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )
    dataloader_spot = DataLoader(
        spot_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_spot, len(spot_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["clevr"] = {"train":dataloader_clevr_train, "val":dataloader_clevr_test, "test":dataloader_clevr_test}
DATALOADER_DICT["spot"] = {"train":dataloader_spot_train, "val":dataloader_spot_test, "test":dataloader_spot_test}
