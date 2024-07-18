import torch, typing
from torch import nn
from agent.baseModule import baseModule

class MAB(baseModule):
    '''
        Multihead attention block
    '''
    def __init__(self, n_feature:int, num_heads:int, fc_hiddens:typing.List[int]) -> None:
        super().__init__()
        self.n_feature = n_feature
        self.n_output = n_feature
        self.num_heads = num_heads
        self.mhAtt = nn.MultiheadAttention(embed_dim=n_feature, num_heads=num_heads, batch_first=True)
        self.layerNorm0 = nn.LayerNorm(n_feature)
        fc = []
        for i in range(len(fc_hiddens)):
            fc.append(nn.Linear(n_feature if i == 0 else fc_hiddens[i-1], fc_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(fc_hiddens[-1], n_feature))
        self.rFC = nn.Sequential(*fc)
        '''
            row-wise feedforward network
        '''
        self.layerNorm1 = nn.LayerNorm(n_feature)

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        H = self.layerNorm0(self.mhAtt(x, y, y)[0]+x)
        z = self.layerNorm1(self.rFC(H)+H)
        return z

class SAB(MAB):
    '''
        Set attention block
    '''
    def __init__(self, n_feature:int, num_heads:int, fc_hiddens:typing.List[int]) -> None:
        super().__init__(n_feature, num_heads, fc_hiddens)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return super().forward(x, x)
    
class ISAB(MAB):
    '''
        Induced set attention block
    '''
    def __init__(self, n_feature:int, num_heads:int, fc_hiddens:typing.List[int], n_induce:int) -> None:
        super().__init__(n_feature, num_heads, fc_hiddens)
        self.n_induce = n_induce
        self.Induce = nn.Parameter(torch.randn(1, n_induce, n_feature))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return super().forward(x, self.Induce.tile((batch_size,1,1)))
    
class PMA(baseModule):
    '''
        Pooling by Multihead Attention
    '''
    def __init__(self, 
                 n_feature:int, fc_hiddens:typing.List[int],
                 n_output:int, num_heads:int, mab_fc_hiddens:torch.List[int],
                 n_seed=1, flat_seed=True) -> None:
        super().__init__()
        self.n_feature = n_feature
        self.n_output = n_output
        self.n_seed = n_seed
        fc = []
        for i in range(len(fc_hiddens)):
            fc.append(nn.Linear(n_feature if i == 0 else fc_hiddens[i-1], fc_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(fc_hiddens[-1], n_output))
        self.rFC = nn.Sequential(*fc)
        '''
            row-wise feedforward network
        '''
        self.mab = MAB(n_output, num_heads, mab_fc_hiddens)
        self.Seed = nn.Parameter(torch.randn(1, n_seed, n_output))
        self.flat_seed = flat_seed

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        z = self.rFC(x)
        pmaz =  self.mab(self.Seed.tile((batch_size,1,1)), z) # shape: (batch_size, n_seed, n_output)
        if self.flat_seed:
            pmaz = pmaz.view(batch_size, self.n_seed*self.n_output)
        return pmaz
    
class setTransformer(baseModule):
    '''
        Set transformer
    '''
    def __init__(self, 
                 n_feature:int, num_heads:int, encoder_fc_hiddens:typing.List[int], encoder_depth:int,
                 n_output:int, pma_fc_hiddens:typing.List[int], pma_mab_fc_hiddens:typing.List[int],
                 final_fc_features:int|None=None,
                 final_fc_hiddens:typing.List[int]|None=None,
                 n_induce=None, 
                 n_seed=1, flat_seed=True) -> None:
        super().__init__()
        self.n_feature = n_feature
        self.n_output = n_output
        self.encoder_depth = encoder_depth
        self.encoder = nn.Sequential()
        for i in range(encoder_depth):
            if n_induce is None:
                self.encoder.add_module(f'SAB{i}', SAB(n_feature, num_heads, encoder_fc_hiddens))
            else:
                self.encoder.add_module(f'SAB{i}', ISAB(n_feature, num_heads, encoder_fc_hiddens, n_induce))
        if (final_fc_features is not None) and (final_fc_hiddens is not None):
            pma = PMA(n_feature, pma_fc_hiddens, final_fc_features, num_heads, pma_mab_fc_hiddens, n_seed=n_seed, flat_seed=flat_seed)
            ffc = []
            for i in range(len(final_fc_hiddens)):
                ffc.append(nn.Linear(final_fc_features if i == 0 else final_fc_hiddens[i-1], final_fc_hiddens[i]))
                ffc.append(nn.ReLU())
            ffc.append(nn.Linear(final_fc_hiddens[-1], n_output))
            self.decoder = nn.Sequential(pma, *ffc)
        else:
            pma = PMA(n_feature, pma_fc_hiddens, n_output, num_heads, pma_mab_fc_hiddens, n_seed=n_seed, flat_seed=flat_seed)
            self.decoder = pma

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x)