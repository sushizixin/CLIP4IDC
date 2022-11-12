import os
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
from skimage import exposure
import matplotlib.pyplot as plt

from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4IDC
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import DATALOADER_DICT


def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4IDC.from_pretrained(args.cross_model, args.decoder_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=4, help='')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=8, help='batch size eval')

    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')

    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="caption", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="clevr", type=str, help="Point the dataset to finetune.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--intra_num_hidden_layers', type=int, default=9, help="Layer NO. of intra module")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")

    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    args = parser.parse_args()

    return args


def load_image(path):
    image = Image.open(path)
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.array(image)
    return image


def visualize(visual_weights, bef_image_paths, aft_image_paths):
    bef_image_path = bef_image_paths
    aft_image_path = aft_image_paths

    bef_image = load_image(bef_image_path)
    aft_image = load_image(aft_image_path)

    last_weights = visual_weights[-1][0]

    ## Choose any weights you would like to see
    ## CLS Token of the first image
    bef_weights, aft_weights = last_weights[0, 1:50], last_weights[0, 51:]
    ## CLS Token of the second image
    # bef_weights, aft_weights = last_weights[50, 1:50], last_weights[50, 51:]

    bef_weights = bef_weights.data.cpu().numpy().reshape(7, 7)
    aft_weights = aft_weights.data.cpu().numpy().reshape(7, 7)

    cam_bef_weights = exposure.rescale_intensity(bef_weights, out_range=(0, 255))
    cam_bef_weights = cv2.resize(cam_bef_weights.astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC)

    cam_aft_weights = exposure.rescale_intensity(aft_weights, out_range=(0, 255))
    cam_aft_weights = cv2.resize(cam_aft_weights.astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC)

    sep_weights = visual_weights[8][0]
    sep_bef_weights, sep_aft_weights = sep_weights[0, 0, 1:], sep_weights[1, 0, 1:]
    sep_bef_weights = sep_bef_weights.data.cpu().numpy().reshape(7, 7)
    sep_aft_weights = sep_aft_weights.data.cpu().numpy().reshape(7, 7)

    cam_sep_bef_weights = exposure.rescale_intensity(sep_bef_weights, out_range=(0, 255))
    cam_sep_bef_weights = cv2.resize(cam_sep_bef_weights.astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC)

    cam_sep_aft_weights = exposure.rescale_intensity(sep_aft_weights, out_range=(0, 255))
    cam_sep_aft_weights = cv2.resize(cam_sep_aft_weights.astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.imshow(bef_image)
    ax1.axis("off")
    ax4.imshow(aft_image)
    ax4.axis("off")

    ax2.imshow(bef_image)
    ax2.imshow(cam_sep_bef_weights, alpha=0.6, cmap="jet")
    ax2.axis("off")

    ax3.imshow(bef_image)
    ax3.imshow(cam_bef_weights, alpha=0.6, cmap="jet")
    ax3.axis("off")

    ax5.imshow(aft_image)
    ax5.imshow(cam_sep_aft_weights, alpha=0.6, cmap="jet")
    ax5.axis("off")

    ax6.imshow(aft_image)
    ax6.imshow(cam_aft_weights, alpha=0.6, cmap="jet")
    ax6.axis("off")

    ## TODO Save Image
    return


def reshape_input(video):
    video = torch.as_tensor(video).float()
    b, pair, channel, h, w = video.shape
    video = video.view(b * pair, channel, h, w)
    video_frame = pair
    return video, video_frame


def visualize_epoch(args, model, test_dataloader, device):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    ## Total numbers of image pairs in the dataset
    data_size = test_dataloader.dataset.sample_len

    ## Choose the index of the image pair
    data_index = 0
    batch = test_dataloader.dataset[data_index]

    image_names = batch[-1]

    if args.datatype == "clevr":
        bef_image, aft_image, nc_image = torch.from_numpy(batch[3]).to(device), torch.from_numpy(batch[4]).to(device), torch.from_numpy(batch[5]).to(device)
        bef_image = bef_image.unsqueeze(1)
        nc_image = nc_image.unsqueeze(1)
        nc_video = torch.cat([bef_image, nc_image], 1)
    else:
        bef_image, aft_image = torch.from_numpy(batch[3]).to(device), torch.from_numpy(batch[4]).to(device)
        bef_image = bef_image.unsqueeze(1)

    aft_image = aft_image.unsqueeze(1)
    video = torch.cat([bef_image, aft_image], 1)

    if args.datatype == "clevr":
        bef_image_paths = os.path.join(args.features_path, "images", "CLEVR_default_%s" % image_names)
        aft_image_paths = os.path.join(args.features_path, "sc_images", "CLEVR_semantic_%s" % image_names)
    elif args.datatype == "spot":
        bef_image_paths = os.path.join(args.features_path, image_names)
        aft_image_paths = os.path.join(args.features_path, image_names.replace(".png", "_2.png"))

    with torch.no_grad():
        video, video_frame = reshape_input(video)
        _, attn_weights = model.clip.visual(video.type(model.clip.dtype), video_frame=video_frame, visualize=True)

        visualize(attn_weights, bef_image_paths, aft_image_paths)
        if args.datatype == "clevr":
            nc_video, nc_video_frame = reshape_input(nc_video)
            _, nc_attn_weights = model.clip.visual(nc_video.type(model.clip.dtype), video_frame=nc_video_frame, visualize=True)
            aft_image_paths = os.path.join(args.features_path, "nsc_images", "CLEVR_nonsemantic_%s" % image_names)
            visualize(nc_attn_weights, bef_image_paths, aft_image_paths)


if __name__ == '__main__':
    args = get_args()

    args.data_path = "your_data/clevr_change/data"
    args.features_path = "your_data/clevr_change/data"

    ## clevr or spot
    args.datatype = "clevr"

    ## Load either pretrained (Retrieval) or trained (Caption) model
    args.init_model = "ckpts/trained/pytorch_model.bin.clevr"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ClipTokenizer()
    model = init_model(args, device, n_gpu=1, local_rank=0)

    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    with torch.no_grad():
        visualize_epoch(args, model, test_dataloader, device)
