# Adapted from ./multi_eval.py, used for program prediction
#
# python==3.6.9
import torch # 1.7.1
import yaml
import pandas as pd
import torchvision.transforms as transforms # 0.8.2
from PIL import Image
import torch.nn as nn
import math
import logging
import os
from typing import Optional

pathPwd = os.getcwd()
pathPwd = os.path.join(pathPwd, 'ePre') #./ePre
print('model_pre:pathpwd:', pathPwd)
print('model_pre:start')
# exit()
# Model architecture code
logger = logging.getLogger(__name__)


# position wise encoding
class PositionalEncodingComponent(nn.Module):
    '''
    Class to encode positional information to tokens.
    For future, I want that this class to work even for sequences longer than 5000
    '''

    def __init__(self, hid_dim, dropout=0.2, max_len=5000):
        super().__init__()

        assert hid_dim % 2 == 0  # If not, it will result error in allocation to positional_encodings[:,1::2] later

        self.dropout = nn.Dropout(dropout)

        self.positional_encodings = nn.Parameter(torch.zeros(1, max_len, hid_dim), requires_grad=False)
        # Positional Embeddings : [1,max_len,hid_dim]

        pos = torch.arange(0, max_len).unsqueeze(1)  # pos : [max_len,1]
        div_term = torch.exp(-torch.arange(0, hid_dim, 2) * math.log(
            10000.0) / hid_dim)  # Calculating value of 1/(10000^(2i/hid_dim)) in log space and then exponentiating it
        # div_term: [hid_dim//2]

        self.positional_encodings[:, :, 0::2] = torch.sin(pos * div_term)  # pos*div_term [max_len,hid_dim//2]
        self.positional_encodings[:, :, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        # TODO: update this for very long sequences
        x = x + self.positional_encodings[:, :x.size(1)].detach()
        return self.dropout(x)


# feed forward
class FeedForwardComponent(nn.Module):
    '''
    Class for pointwise feed forward connections
    '''

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)

    def forward(self, x):
        # x : [batch_size,seq_len,hid_dim]
        x = self.dropout(torch.relu(self.fc1(x)))

        # x : [batch_size,seq_len,pf_dim]
        x = self.fc2(x)

        # x : [batch_size,seq_len,hid_dim]
        return x


# multi headed attention
class MultiHeadedAttentionComponent(nn.Module):
    '''
    Multiheaded attention Component.
    '''

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0  # Since we split hid_dims into n_heads

        self.hid_dim = hid_dim
        self.n_heads = n_heads  # no of heads in 'multiheaded' attention
        self.head_dim = hid_dim // n_heads  # dims of each head

        # Transformation from source vector to query vector
        self.fc_q = nn.Linear(hid_dim, hid_dim)

        # Transformation from source vector to key vector
        self.fc_k = nn.Linear(hid_dim, hid_dim)

        # Transformation from source vector to value vector
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        # Used in self attention for smoother gradients
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([self.head_dim])), requires_grad=False)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        # query : [batch_size, query_len, hid_dim]
        # key : [batch_size, key_len, hid_dim]
        # value : [batch_size, value_len, hid_dim]

        batch_size = query.shape[0]

        # Transforming query, key, values
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q : [batch_size, query_len, hid_dim]
        # K : [batch_size, key_len, hid_dim]
        # V : [batch_size, value_len,hid_dim]

        # Changing shapes to accommodate n_heads information
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q : [batch_size, n_heads, query_len, head_dim]
        # K : [batch_size, n_heads, key_len, head_dim]
        # V : [batch_size, n_heads, value_len, head_dim]

        # Calculating alpha
        score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # score : [batch_size, n_heads, query_len, key_len]

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e10)

        alpha = torch.softmax(score, dim=-1)
        # alpha : [batch_size, n_heads, query_len, key_len]

        # Get the final self-attention vector
        x = torch.matmul(self.dropout(alpha), V)
        # x : [batch_size, n_heads, query_len, head_dim]

        # Reshaping self-attention vector to concatenate
        x = x.permute(0, 2, 1, 3).contiguous()
        # x : [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x: [batch_size, query_len, hid_dim]

        # Transforming concatenated outputs
        x = self.fc_o(x)
        # x : [batch_size, query_len, hid_dim]

        return x, alpha


# EncodingLayer
class EncodingLayer(nn.Module):
    '''
    Operations of a single layer. Each layer contains:
    1) multihead attention, followed by
    2) LayerNorm of addition of multihead attention output and input to the layer, followed by
    3) FeedForward connections, followed by
    4) LayerNorm of addition of FeedForward outputs and output of previous layerNorm.
    '''

    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after self-attention
        self.ff_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after FeedForward component

        self.self_attention = MultiHeadedAttentionComponent(hid_dim, n_heads, dropout)
        self.feed_forward = FeedForwardComponent(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src : [batch_size, src_len, hid_dim]
        # src_mask : [batch_size, 1, 1, src_len]

        # get self-attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # LayerNorm after dropout
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src : [batch_size, src_len, hid_dim]

        # FeedForward
        _src = self.feed_forward(src)

        # layerNorm after dropout
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src: [batch_size, src_len, hid_dim]

        return src


class AudioRepresentations(nn.Module):
    '''
    Group of layers that give final audio representation for cross attention

    The class get an input of size [batch_size,max_audio_len]
    we split the max_audio_len by audio_split_samples.
    Example: if the input was [10,60000] and audio_split_samples as 1000
    then we split the input as [10,60,1000]
    '''

    def __init__(self, audio_split_samples, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length):
        super().__init__()

        # Used for splitting the original signal
        self.audio_split_samples = audio_split_samples

        # Transform input from audio_split_dim to hid_dim
        self.transform_input = nn.Linear(audio_split_samples, hid_dim)

        self.pos_embedding = PositionalEncodingComponent(hid_dim, dropout, max_length)

        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

    def forward(self, audio):
        # You don't need mask for audio in attention because that padded
        # audio : [batch_size, max_audio_len]

        assert audio.shape[1] % self.audio_split_samples == 0

        batch_size = audio.shape[0]
        audio = audio.reshape(batch_size, -1, self.audio_split_samples)
        # audio : [batch_size, src_len , audio_split_samples]

        audio_embeddings = self.transform_input(audio) * self.scale
        # audio embeddings : [batch_size, src_len, hid_dim]

        # TODO: find better ways to give positional information. Here it is giving each audio_split_sample chunk same
        #  positional embedding
        audio = self.pos_embedding(audio_embeddings)
        # audio : [batch_size, src_len, hid_dim]

        for layer in self.layers:
            audio = layer(audio)
        # audio : [batch_size, src_len, hid_dim]

        return audio


class TextRepresentations(nn.Module):
    """
    Group of layers that give final text representation for cross attention
    """

    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, text_pad_index, max_length=5000):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = PositionalEncodingComponent(hid_dim, dropout, max_length)

        # encoder layers
        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

        self.text_pad_index = text_pad_index

    def create_text_mask(self, text):
        # masks padded values of text

        # text : [batch_size, src_len]
        text_mask = (text != self.text_pad_index).unsqueeze(1).unsqueeze(2)

        return text_mask

    def forward(self, text):
        # text : [batch_size, src_len]

        text_mask = self.create_text_mask(text)
        # text_mask : [batch_size,1,1,src_len]

        batch_size = text.shape[0]
        src_len = text.shape[1]

        tok_embeddings = self.tok_embedding(text) * self.scale

        # token plus position embeddings
        text = self.pos_embedding(tok_embeddings)

        for layer in self.layers:
            text = layer(text, text_mask)
        # src : [batch_size, src_len, hid_dim]

        return text


class ImageRepresentations(nn.Module):
    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, dropout, channels, pool='cls',
                 image_size=224, patch_size=16, flag=1, **kwargs):
        super().__init__()
        self.flag = flag
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch = lambda x: x.view(-1, channels, image_width // patch_width, patch_width, \
                                         image_height // patch_height, patch_height).permute(0, 2, 4, 1, 3, 5).reshape(
            -1, num_patches, patch_dim)
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim)
        )
        self.pos_embedding = PositionalEncodingComponent(hid_dim, dropout, num_patches + 1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.dropout = nn.Dropout(dropout)

        # encoder layers
        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.to_latent = nn.Identity()

        self.hid_dim = hid_dim
        self.pool = pool

    def forward(self, image):
        x = self.to_patch(image)
        x = self.to_patch_embedding(x)
        # torch.save(self.to_patch_embedding, "./patch")
        # if self.flag==1:
        #    print(x[:,5,:])
        #    print(x[0,5,:])
        #    print(x[10,5,:])
        #    print(next(self.to_patch_embedding[1].parameters()))
        batch_size, n, _ = x.shape
        cls_tokens = self.cls_token.expand(batch_size, 1, self.hid_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.to_latent(x)


# Cross Attention Layer
class CrossAttentionLayer(nn.Module):
    '''
    This layer takes input the audio and text representations after they have been
    passed through their respective Encoding layers.
    The text representations will act as query
    the audio representations will be key and values.
    So this will take most important features from text representation based on the
    attention between audio and the text features.
    '''

    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after self-attention
        self.ff_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after FeedForward component

        self.self_attention = MultiHeadedAttentionComponent(hid_dim, n_heads, dropout)
        self.feed_forward = FeedForwardComponent(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, audio):
        # text : [batch_size, text_len, hid_dim]
        # audio : [batch_size, audio_len, hid_dim

        # get self-attention
        _text, _ = self.self_attention(text, audio, audio)  # query, key, value

        # LayerNorm after dropout
        text = self.self_attn_layer_norm(text + self.dropout(_text))
        # text : [batch_size, text_len, hid_dim]

        # FeedForward
        _text = self.feed_forward(text)

        # layerNorm after dropout
        text = self.ff_layer_norm(text + self.dropout(_text))
        # text: [batch_size, text_len, hid_dim]

        return text


# Model
class Model(nn.Module):
    """
    Model class
    We will use <sos> token for prediction of classes
    """

    def __init__(self, audio_split_samples, hid_dim, audio_representation_layers, n_heads, pf_dim, dropout, max_length \
                 , len_text_vocab, text_pad_index, text_representation_layers, image_representation_layers, \
                 channels1, image_size1, patch_size1, channels2, image_size2, patch_size2, cross_attention_layers, \
                 output_dim, config):
        super().__init__()
        self.audio_representations = AudioRepresentations(audio_split_samples, hid_dim, audio_representation_layers,
                                                          n_heads, pf_dim, dropout, max_length)
        self.text_representations = TextRepresentations(len_text_vocab, hid_dim, text_representation_layers, n_heads,
                                                        pf_dim, dropout, text_pad_index, max_length)

        self.image_representations1 = ImageRepresentations(hid_dim, image_representation_layers, n_heads, pf_dim,
                                                           dropout,
                                                           channels1, image_size=image_size1, patch_size=patch_size1)

        self.image_representations2 = ImageRepresentations(hid_dim, image_representation_layers, n_heads, pf_dim,
                                                           dropout, channels2, image_size=image_size2,
                                                           patch_size=patch_size2)

        self.cross_attention = [[CrossAttentionLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(4)] \
                                for _ in range(cross_attention_layers)]

        self.feed_forward = nn.Sequential(
            nn.Linear(hid_dim, int(hid_dim / 2)),
            nn.BatchNorm1d(int(hid_dim / 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hid_dim / 2), output_dim),
            nn.ReLU()
        )

        self.loss = nn.L1Loss()

        self.config = config

    def forward(self, audio, text, image1, image2, label):
        # audio : [batch_size, max_audio_len]
        # text : [batch_size, src_len]

        audio = self.audio_representations(audio)
        # audio : [batch_size, audio_len, hid_dim] where audio_len= max_audio_len/audio_split_samples

        text = self.text_representations(text)
        # text : [batch_size, src_len, hid_dim]

        image1 = self.image_representations1(image1)
        # image1: [batch_size, num_patches, w, h]

        image2 = self.image_representations2(image2)
        # image2: [batch_size, num_patches, w, h]

        for layer in self.cross_attention:
            text = layer[0](text, image2)
            image1 = layer[1](image1, text)
            audio = layer[2](audio, image1)
            image2 = layer[3](image2, audio)

        pred_token = image2[:, 0, :]
        # pred_token : [batch_size, hid_dim]

        output = self.feed_forward(pred_token)

        loss = self.loss(output, label)

        return {'loss': loss, "predicted_y": output}


class ModelSatelliteScene(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, image_representation_layers, \
                 channels1, image_size1, patch_size1, channels2, image_size2, patch_size2, cross_attention_layers, \
                 output_dim, config):
        super().__init__()

        self.image_representations1 = ImageRepresentations(hid_dim, image_representation_layers, n_heads, pf_dim,
                                                           dropout,
                                                           channels1, image_size=image_size1, patch_size=patch_size1)

        self.image_representations2 = ImageRepresentations(hid_dim, image_representation_layers, n_heads, pf_dim,
                                                           dropout, channels2, flag=2, image_size=image_size2,
                                                           patch_size=patch_size2)

        # self.cross_attention = nn.ModuleList(
        #    [CrossAttentionLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(cross_attention_layers)])

        # self.feed_forward = nn.Sequential(
        #     nn.BatchNorm1d(hid_dim),
        #     nn.Linear(hid_dim, int(hid_dim / 2)),
        #     nn.BatchNorm1d(int(hid_dim / 2)),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(int(hid_dim / 2), output_dim)
        #     )
        self.loss = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.config = config

        self.dnn = nn.Sequential(
            nn.BatchNorm1d(2 * hid_dim),
            nn.Linear(2 * hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, output_dim)
        )
        self.cross_attention1 = nn.ModuleList(
            [CrossAttentionLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(cross_attention_layers)])
        self.cross_attention2 = nn.ModuleList(
            [CrossAttentionLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(cross_attention_layers)])

    def forward(self, image1, image2, label):
        image1 = self.image_representations1(image1)
        # print(image1[:,0,:])
        # image1: [batch_size, num_patches, w, h]
        image2 = self.image_representations2(image2)
        # print(image2[:,0:,])
        # image2: [batch_size, num_patches, w, h]
        for layer1, layer2 in zip(self.cross_attention1, self.cross_attention2):
            image1 = layer1(image1, image2)
            image2 = layer2(image2, image1)
        # image1 = image1[:, 0, :].view(-1, image1.shape[2])
        # image2 = image2[:, 0, :].view(-1, image2.shape[2])
        image1 = image1.mean(dim=1)
        image2 = image2.mean(dim=1)
        output = self.dnn(torch.cat([image1, image2], dim=1))

        # for layer in self.cross_attention:
        #     image1 = layer(image1, image2)
        # pred_token = image2[:, 0, :]
        # pred_token : [batch_size, hid_dim]
        # output = self.feed_forward(pred_token).reshape(-1)

        loss = self.loss(output.reshape(-1), label.reshape(-1).float())
        return {'loss': loss, "predicted_y": output,
                "l1": self.l1(output.reshape(-1), label.reshape(-1).float()).item()}

# paths
# path = '/users/apple/downloads/all-ali/'
path =  pathPwd+ '/model/'
model_name ='multimodal-multilabel-model_0910.pt'
# images_match_path = 'images_match的副本.csv'
config_path = 'config.yaml'

# config and model
f = open(path + config_path, 'r', encoding='utf-8')
cont = f.read()
config = yaml.safe_load(cont)
model = ModelSatelliteScene(
    n_heads = config['n_heads'],
    hid_dim = config['hid_dim'],
    image_size1 = config['image_size1'],
    output_dim = 1,
    pf_dim = config['pf_dim'],
    dropout = config['dropout'],
    image_representation_layers = config['image_representation_layers'],
    channels1 = config['channels1'],
    patch_size1 = config['patch_size1'],
    channels2 = config['channels2'],
    image_size2 = config['image_size2'],
    patch_size2 = config['patch_size2'],
    cross_attention_layers = config['cross_attention_layers'],
    config = config)
model.load_state_dict(torch.load(path + model_name, map_location=torch.device('cpu')))

# Import already matched street view and satellite images by latitude and longitude coordinates
# images_match = pd.read_csv(path + images_match_path)

# eval mode
model.eval()

# View the first two sets of images, export and re-establish the sample image folder 'sat_sce_pics'
# print('iloc:',images_match.iloc[:2,:])
# print('y:',images_match.iloc[:2,:]['y']) # View night light value, i.e., column y

# y = images_match.iloc[:2,:]['y']

# Preprocessing Transform for street view and satellite images, mean and standard deviation are labeled below
def sat_transform(sat_img_dir):
    sat_img = Image.open(sat_img_dir)
    trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4080, 0.4171, 0.3150],
                                     std=[0.1742, 0.1433, 0.1419])])
    tr_sat = trans(sat_img)
    return tr_sat

def sce_transform(sce_img_dir):
    sce_img = Image.open(sce_img_dir)
    trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5610, 0.5693, 0.5843],
                                     std=[0.2690, 0.2647, 0.2991])])
    tr_sce = trans(sce_img)
    return tr_sce

# Input model, folder path, street view path, satellite path, and y (float, changed to torch.tensor in the code)
# Parameter y: used to calculate l1 (absolute error between predicted value and actual value), input 0 during prediction, does not affect prediction results, but l1 is unreliable in this case
def get_outputs(usedmodel, folder_dir, sat_img_dir, sce_img_dir, y):
    tr_sat = sat_transform(folder_dir + sat_img_dir)
    tr_sce = sce_transform(folder_dir + sce_img_dir)
    output = usedmodel(tr_sat , tr_sce, torch.tensor(y)) # Model input: transformed satellite, transformed street view, 100 times the night light
    return output

# Run under sample folder
# for i in range(2):
#     folder_dir =pathPwd+ '/sat_sce_pics/'
#     sat_img_name = 'sat_' + str(i+1) + '.png'
#     sce_img_name = 'sce_' + str(i+1) + '.png'
#     output = get_outputs(model, folder_dir, sat_img_name, sce_img_name,  0 )
#     print(output)

def extPre(satImgPath,secImgPath):
    try:
        tr_sat = sat_transform(satImgPath)
        # d_numpy=tr_sat.numpy()
        # print('type:',type(d_numpy))
        # print('shape:',d_numpy.shape)
        tr_sce = sce_transform(secImgPath)
        # d_numpy=tr_sce.numpy()
        # print('type:',type(d_numpy))
        # print('shape:',d_numpy.shape)
        # print(d_numpy)
        # return 0
        output = model(tr_sat , tr_sce, torch.tensor(0))
        # {'loss': tensor(1520.1289, grad_fn=<MseLossBackward>), 'predicted_y': tensor([[38.9888]], grad_fn=<AddmmBackward>), 'l1': 38.98883056640625}
        score=output.get('predicted_y').tolist()[0][0]
    except Exception as e:
       print('Error executing calculation, e:', e)
       return 0
    return score

def mainExt():
    satImgPath=os.path.join(pathPwd, 'data_img2','51RUQ_18_17_121.4338896959315_31.19128964591694.jpg')
    # satImgPath=os.path.join(pathPwd, 'sat_sce_pics','sat_1.png')
    secImgPath=os.path.join(pathPwd,'sat_sce_pics','sce_1.png')
    secImgPath=os.path.join(pathPwd,'sat_sce_pics','default.jpg')
    # secImgPath=os.path.join(pathPwd,'sat_sce_pics','default_s.png')
    # secImgPath='./data/temp/data_scene02/sce_pics/50SPA/118.639162_32.074757_90_0.png' # A file that will report an error
    return extPre(satImgPath,secImgPath)

if __name__ == '__main__':
    res= mainExt()
    print("res:", res)
    # print("test2",res.get('loss').tolist())
    # print("test2",type(res.get('loss')))
    # print("test2",res.get('predicted_y').tolist()[0][0])
    # print("test2",res.get('l1'))
