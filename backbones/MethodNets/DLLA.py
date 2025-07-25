import torch.nn as nn

class DLLAModel(nn.Module):

    def __init__(self, args, backbone):

        super(DLLAModel, self).__init__()
        self.backbone = backbone
        activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
        args.feat_dim = args.base_dim
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.mlp_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(args.base_dim, args.base_dim), 
        )
        

    def forward(self, text,text_keyword,text_keyframe, video, audio, mode='train'):
        
        if mode == 'pretrain-mm':
            features = self.backbone(text,text_keyword,text_keyframe, video, audio)
            mlp_output = self.mlp_head(features)
            return mlp_output

        elif mode == 'train-mm':
            features = self.backbone(text, text_keyword,text_keyframe,video, audio)
            mlp_output = self.mlp_head(features)

            return features, mlp_output
                