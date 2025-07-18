
class Param():
    
    def __init__(self, args):

        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        if args.multimodal_method == 'dlla':
            hyper_parameters = {
                # 'pretrained_bert_model': 'uncased_L-12_H-768_A-12',
                'pretrained_bert_model': '/root/autodl-tmp/uncased_L-12_H-768_A-12',
                'pretrain_batch_size': 128,
                'train_batch_size': 128,
                'eval_batch_size': 128,
                'test_batch_size': 128,
                'num_pretrain_epochs': 100,
                'num_train_epochs': 100,
                'pretrain': [False],
                'aligned_method': 'ctc',
                'need_aligned': False,
                'freeze_pretrain_bert_parameters': [True],
                'freeze_train_bert_parameters': [True],
                'pretrain_temperature': [0.1],
                'train_temperature_sup': [7.5],     
                'train_temperature_unsup': [0.9],    
                'activation': 'tanh',
                'lr_pre': [1e-5],
                'lr': [3e-4],     
                'delta': [0.05],
                'thres': [0.1],
                'topk': [5],
                'weight_decay': 0.01,
                'feat_dim': 768,
                'hidden_size': 768,
                'grad_clip': -1.0,
                'warmup_proportion': 0.1,
                'hidden_dropout_prob': 0.01,
                'weight': 1.0,
                'loss_mode': 'rdrop',
                'base_dim': 256,
                'nheads': 8,
                'attn_dropout': 0.1,
                'relu_dropout': [0.01],
                'embed_dropout': [0.01],
                'res_dropout': 0.0,
                'attn_mask': True,
                'encoder_layers_1': 1,
                'fusion_act': 'tanh',
                'cross_num_heads': 8,
                'cross_dp_rate': [0.1],
                'n_levels_cross': 2,
                'dropout_rate': [0.01],
                'self_num_heads': 8,
                'n_levels_self': 1,
            }
        else:
            print('Not Supported Multimodal Method')
            raise NotImplementedError
            
        return hyper_parameters
    