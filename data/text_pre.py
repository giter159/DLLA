import os
import csv
import sys
from transformers import BertTokenizer, RobertaTokenizer

def get_t_data(args, data_args, key = None):
    
    if args.text_backbone.startswith('bert'):
        t_data = get_data(args, data_args, key)
    else:
        raise Exception('Error: inputs are not supported text backbones.')

    return t_data

def get_data(args, data_args, key):

    processor = DatasetProcessor(args, key)
    data_path = data_args['data_path']
    if key is not None:
        if args.dataset == 'MELD-DA':
            data_path = "/root/autodl-tmp/autodl/datasets/key_MELD"
        if args.dataset == 'IEMOCAP-DA':
            data_path = "/root/autodl-tmp/autodl/datasets/key_IEMOCAP"
        if args.dataset == 'MIntRec':
            data_path = "/root/autodl-tmp/autodl/datasets/key_MIntRec"

    train_examples = processor.get_examples(data_path, 'train') 
    dev_examples = processor.get_examples(data_path, 'dev')

    train_examples = train_examples + dev_examples

    train_feats = get_backbone_feats(args, data_args, train_examples)

    
    test_examples = processor.get_examples(data_path, 'test')
    test_feats = get_backbone_feats(args, data_args, test_examples)

    
    outputs = {
        'train': train_feats,
        'test': test_feats
    }
        
    return outputs

def get_backbone_feats(args, data_args, examples):
    
    if args.text_backbone.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)   
   
    features = convert_examples_to_features(examples, args.text_seq_len, tokenizer)     
    features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]

    return features_list

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def __init__(self, args, key):
        super(DatasetProcessor).__init__()

        if key == 'keyframe':
            if args.dataset in ['MIntRec']:
                self.select_id = 4
            elif args.dataset in ['clinc', 'clinc-small', 'snips', 'atis']:
                self.select_id = 1
            elif args.dataset in ['L-MIntRec']:
                self.select_id = 6
            elif args.dataset in ['MELD-DA']:
                self.select_id = 3
            elif args.dataset in ['IEMOCAP-DA']:
                self.select_id = 2
        else:
            if args.dataset in ['MIntRec']:
                self.select_id = 3
            elif args.dataset in ['clinc', 'clinc-small', 'snips', 'atis']:
                self.select_id = 0
            elif args.dataset in ['L-MIntRec']:
                self.select_id = 5
            elif args.dataset in ['MELD-DA']:
                self.select_id = 2
            elif args.dataset in ['IEMOCAP-DA']:
                self.select_id = 1
        
    def get_examples(self, data_dir, mode):
        
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'all':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "all.tsv")), "all")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[self.select_id]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
        
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()