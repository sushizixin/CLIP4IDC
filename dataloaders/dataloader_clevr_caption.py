from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import random
from dataloaders.rawimage_util import RawImageExtractor
from collections import defaultdict


class CLEVR_DataLoader(Dataset):
    """CLEVR dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.default_features_path = os.path.join(self.data_path, 'images')
        self.nsc_features_path = os.path.join(self.data_path, 'nsc_images')
        self.sc_features_path = os.path.join(self.data_path, 'sc_images')
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        image_id_path = os.path.join(self.data_path, 'splits.json')
        change_caption_file = os.path.join(self.data_path, "change_captions.json")
        no_change_caption_file = os.path.join(self.data_path, "no_change_captions.json")

        with open(image_id_path, 'r') as fp:
            image_ids = json.load(fp)[subset]

        with open(change_caption_file, 'r') as fp:
            change_captions = json.load(fp)

        with open(no_change_caption_file, 'r') as fp:
            no_change_captions = json.load(fp)

        self.no_change_captions = no_change_captions

        image_dict = {}
        image_files = os.listdir(self.default_features_path)
        for image_file in image_files:
            image_id_ = image_file.split(".")[0].split('_')[-1]
            if int(image_id_) not in image_ids:
                continue
            # file_path_ = os.path.join(self.default_features_path, image_file)
            image_dict[int(image_id_)] = image_id_
        self.image_dict = image_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for image_id in image_ids:
            image_id_name = "CLEVR_default_%s.png" % self.image_dict[image_id]
            assert image_id_name in change_captions
            self.sentences_dict[len(self.sentences_dict)] = (image_id, change_captions[image_id_name])
            # for cap_txt in change_captions[image_id_name]:
            #     self.sentences_dict[len(self.sentences_dict)] = (image_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        # self.nc_sentences_dict = defaultdict(list)
        # for image_id in image_ids:
        #     image_id_name = "CLEVR_default_%s.png" % self.image_dict[image_id]
        #     assert image_id_name in no_change_captions
        #     for cap_txt in no_change_captions[image_id_name]:
        #         self.nc_sentences_dict[image_id].append(cap_txt)

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.image_num: used to cut the image pair representation
        self.multi_sentence_per_pair = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.image_num = len(image_ids)
            assert len(self.cut_off_points) == self.image_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, image number: {}".format(self.subset, self.image_num))

        print("Image number: {}".format(len(self.image_dict)))
        print("Total Paire: {}".format(len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawImageExtractor = RawImageExtractor(size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, image_id, caption):
        k = 1
        choice_image_ids = [image_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i, image_id in enumerate(choice_image_ids):
            words = []

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + caption_words
            output_caption_words = caption_words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids

    def _get_rawimage(self, image_path):
        choice_image_path = [image_path]
        # Pair x L x T x 3 x H x W
        image = np.zeros((len(choice_image_path), 3, self.rawImageExtractor.size,
                          self.rawImageExtractor.size), dtype=np.float)

        for i, image_path in enumerate(choice_image_path):

            raw_image_data = self.rawImageExtractor.get_image_data(image_path)
            raw_image_data = raw_image_data['image']

            image[i] = raw_image_data

        return image

    def __getitem__(self, idx):
        image_id, caption = self.sentences_dict[idx]
        caption = random.choice(caption)
        image_name = "CLEVR_default_%s.png" % self.image_dict[image_id]
        image_idx_name = "%s.png" % self.image_dict[image_id]

        bef_image_path = os.path.join(self.default_features_path, image_name)
        aft_image_path = os.path.join(self.sc_features_path, image_name.replace('default', 'semantic'))
        no_image_path = os.path.join(self.nsc_features_path, image_name.replace('default', 'nonsemantic'))

        pairs_text, pairs_mask, pairs_segment, pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = self._get_text(image_id, caption)

        no_caption = self.no_change_captions[image_name]
        no_caption = random.choice(no_caption)

        _, _, _, no_pairs_input_caption_ids, no_pairs_decoder_mask, no_pairs_output_caption_ids = self._get_text(image_id, no_caption)

        bef_image = self._get_rawimage(bef_image_path)
        aft_image = self._get_rawimage(aft_image_path)
        no_image = self._get_rawimage(no_image_path)

        image_mask = np.ones(2, dtype=np.long)
        return pairs_text, pairs_mask, pairs_segment, bef_image, aft_image, no_image, image_mask, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
               no_pairs_input_caption_ids, no_pairs_decoder_mask, no_pairs_output_caption_ids, image_idx_name
