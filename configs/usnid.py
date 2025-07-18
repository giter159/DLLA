
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

        if args.multimodal_method == 'text':
            hyper_parameters = {
                'pretrained_bert_model': '/root/autodl-tmp/uncased_L-12_H-768_A-12',
                'num_pretrain_epochs': 100,
                'num_train_epochs': 100,
                'pretrain': True,
                'freeze_pretrain_bert_parameters': True,
                'freeze_train_bert_parameters': True,
                'feat_dim': 768,
                'hidden_size': 768,
                'warmup_proportion': 0.1,
                'lr_pre': 2e-5,
                'lr': 5e-5,
                'weight_decay': 0.01,
                'hidden_dropout_prob': 0.1,
                'pretrain_temperature': 0.07,
                'train_temperature': 0.07,
                're_prob': 0.5,
                'activation': 'tanh',
                'tol': 0.0005,
                'grad_clip': 1.0,
                'train_batch_size': 128,
                'pretrain_batch_size': 128,
                'eval_batch_size': 64,
                'test_batch_size': 64,
            }
        
        else:
            print('Not Supported Multimodal Method')
            raise NotImplementedError
            
        return hyper_parameters