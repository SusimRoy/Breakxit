import torch
import torchvision
import torch.nn as nn
from memorywrap import MemoryWrapLayer
from models.SeqAttention import SeqAttention


class Mapping(nn.Module):
    def __init__(self, args, backbone, n_cls):
        super(Mapping, self).__init__()

        self.atten1 = SeqAttention(64*1, 32*1, 1, args.sqa_type, args.residual_mode)
        self.atten2 = SeqAttention(160*1, 16*1, 2, args.sqa_type, args.residual_mode)
        self.atten3 = SeqAttention(320*1, 64*1, 2, args.sqa_type, args.residual_mode)
        self.atten4 = SeqAttention(640*1, 256*1, 1, args.sqa_type, args.residual_mode)

        self.memory1 = MemoryWrapLayer(32, 128, classifier=None, distance='cosine')
        self.memory2 = MemoryWrapLayer(128, 256, classifier=None, distance='cosine')
        self.memory3 = MemoryWrapLayer(256, 640, classifier=None, distance='cosine')

        self.fc = nn.Linear(640+n_cls,n_cls)
        # self.fc = nn.Linear(128,n_cls)


        self.model = backbone

    def vectorization(self, x):
        f,x = self.model(x, is_feat = True)
        a1 = self.atten1(f[0])
        a2 = self.atten2(f[1])
        a3 = self.atten3(f[2])
        a4 = self.atten4(f[3])
        b = a1.shape[0]
        a1,a2,a3,a4 = a1.view(b, -1), a2.view(b, -1), a3.view(b, -1), a4.view(b, -1)
        # print(a1.shape,a2.shape,a3.shape,a4.shape)
        conc1 = self.memory1(a1, a2)
        conc2 = self.memory2(conc1, a3)
        conc3 = self.memory3(conc2, a4)

        return x, conc3
    
    def joining(self, x, feat_vec):

        return torch.cat((x, feat_vec), dim = 1)

    def forward(self, x):
        x , feat_vec = self.vectorization(x)
        final_vector = self.joining(x , feat_vec)
        # final_vector = feat_vec
        final_vector = self.fc(final_vector)

        return final_vector


