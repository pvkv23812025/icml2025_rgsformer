import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# def complement_graph(edge_index, num_nodes):
#     """
#     Constructs the complementary graph given the edge index of a graph.
    
#     Args:
#         edge_index (torch.Tensor): Edge indices of the graph (shape: [2, num_edges]).
#         num_nodes (int): Number of nodes in the graph.
    
#     Returns:
#         torch.Tensor: Edge indices of the complementary graph (shape: [2, num_edges_complementary]).
#     """
#     # Create a set of all possible edges in a complete graph
#     complete_edges = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes))
#     complete_edges = complete_edges[complete_edges[:, 0] != complete_edges[:, 1]]  # Remove self-loops

#     # Convert edge_index to a set for quick lookup
#     edge_set = set(tuple(edge.cpu().numpy()) for edge in edge_index.t())

#     # Identify complementary edges by checking absence in the existing edge set
#     complement_edges = [tuple(edge.tolist()) for edge in complete_edges if tuple(edge.tolist()) not in edge_set]

#     # Convert the complementary edges back to a tensor
#     complement_edge_index = torch.tensor(complement_edges, dtype=torch.long).t().contiguous()
    
#     return complement_edge_index

def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    #qs,_ = torch.linalg.qr(qs)
    #ks,_ = torch.linalg.qr(ks)
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention=torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1) #[N, N]
        normalizer=attention_normalizer.squeeze(dim=-1).mean(dim=-1,keepdims=True) #[N,1]
        attention=attention/normalizer


    if output_attn:
        return attn_output, attention
    else:
        return attn_output


def geodesic_distance(q, k):
    # Implement geodesic distance for the manifold, e.g., Stiefel
    # For simplicity, assuming q and k are orthonormal matrices
    return torch.norm(torch.matmul(q, k.T) - torch.eye(q.shape[0], device=q.device))


# def full_attention_conv(qs, ks, vs, output_attn=False):
#     """
#     Implements the original scaled dot-product attention mechanism.

#     Args:
#         qs (torch.Tensor): Query matrix of shape [N, H, M].
#         ks (torch.Tensor): Key matrix of shape [L, H, M].
#         vs (torch.Tensor): Value matrix of shape [L, H, D].
#         output_attn (bool): Whether to return attention weights.

#     Returns:
#         attn_output (torch.Tensor): Aggregated attention outputs of shape [N, H, D].
#         attention (torch.Tensor, optional): Attention weights of shape [N, L].
#     """
#     # Scaled dot-product
#     #qs,_ = torch.linalg.qr(qs)
#     #ks,_ = torch.linalg.qr(ks)
#     qs = qs / torch.norm(qs, p=2)  # [N, H, M]
#     ks = ks / torch.norm(ks, p=2)  # [L, H, M]
#     d_k = ks.shape[-1]  # Dimension of key
#     scores = torch.einsum("nhm,lhm->nlh", qs, ks) / (d_k ** 0.5)  # [N, L]
#     #scores = torch.exp(-1*geodesic_distance(qs, ks))
#     #applying geodesic distance each step

#     # Softmax to compute attention weights
#     attention = torch.softmax(scores, dim=-1)  # [N, L]

#     # Weighted sum of values
#     attn_output = torch.einsum("nlh,lhd->nhd", attention, vs)  # [N, H, D]

#     if output_attn:
#         return attn_output, attention
#     else:
#         return attn_output
        
class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)

        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1,
                                             self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1,
                                            self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1,
                                                  self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(
                query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, in_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(in_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(in_channels, in_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(in_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act=use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x = data['graph']['node_feat']
        edge_index = data['graph']['edge_index']
        edge_weight = data['graph']['edge_weight'] if 'edge_weight' in data['graph'] else None
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            if self.use_act:
                x = self.activation(x) 
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]

class ExpMapWithLog(nn.Module):
    def __init__(self):
        super(ExpMapWithLog, self).__init__()
    
    def forward(self, x):
        return torch.exp(x)
    
    def backward(self, grad_output):
        grad_input = grad_output * torch.log(grad_output.clamp(min=1e-10))
        return grad_input

class SGFormer(nn.Module):
    def __init__(self, in_channels, in_gnn_channels, in_embed_features, hidden_channels, out_channels, num_layers=2, num_heads=1, 
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True, use_act=False, graph_weight=0.8, gnn=None, gnnComp=None, aggregate='add'):
        super().__init__()
        self.trans_conv=TransConv(in_channels,out_channels,num_layers,num_heads,alpha,dropout,use_bn,use_residual,use_weight)
        self.gnn=gnn
        self.gnnComp=gnnComp
        self.use_graph=use_graph
        self.graph_weight=graph_weight
        self.use_act=use_act
        self.exp_log=ExpMapWithLog()

        self.aggregate=aggregate

        if aggregate=='add':
            self.fc=nn.Linear(out_channels,out_channels)
        elif aggregate=='cat':
            self.fc=nn.Linear(2*out_channels,out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')
        
        self.params1=list(self.trans_conv.parameters())
        self.params2=list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()) )
         # Projection layers for Q, K, and V
        self.W_q = nn.Linear(in_gnn_channels,in_channels, bias=False)  # Projection for queries
        self.W_k = nn.Linear(in_embed_features,in_channels, bias=False)   # Projection for keys
        self.W_v = nn.Linear(in_gnn_channels,in_channels, bias=False)   # Projection for values
        self.W_p = nn.Linear(in_channels,in_channels,bias=False)
        self.W_r = nn.Linear(in_gnn_channels,in_channels,bias=False)
    # def forward(self,data):
    #     x1=self.trans_conv(data)
    #     if self.use_graph:
    #         x2=self.gnn(data)
    #         if self.aggregate=='add':
    #             x=self.graph_weight*x2+(1-self.graph_weight)*x1
    #         else:
    #             x=torch.cat((x1,x2),dim=1)
    #     else:
    #         x=x1
    #     x=self.fc(x)
    #     return x

    def get_attention_feature(self, gnn_features, embed_features):
        """
        Forward pass for the attention model.
        
        Args:
            x1 (torch.Tensor): Query matrix of shape (batch_size, query_dim).
            x2 (torch.Tensor): Key/Value matrix of shape (batch_size, key_dim).
        
        Returns:
            torch.Tensor: Final output of attention mechanism, shape (batch_size, latent_dim).
        """
        # Project inputs into latent space
        x1 = gnn_features
        x2 = embed_features
        q_proj = self.W_q(x1)  # Shape: (batch_size, latent_dim)
        k_proj = self.W_k(x2)  # Shape: (batch_size, latent_dim)
        #q_proj,_ = torch.linalg.qr(q_proj)
        #k_proj,_ = torch.linalg.qr(k_proj)
        #q_proj = self.exp_log(q_proj)
        #k_proj = self.exp_log(k_proj)
        #q_proj,_ = torch.linalg.qr(q_proj)
        #k_proj,_ = torch.linalg.qr(k_proj)
        v_proj = self.W_v(x1)  # Shape: (batch_size, latent_dim)
        
        # Compute attention scores (scaled dot product)
        scores = torch.matmul(q_proj, k_proj.T) / torch.sqrt(torch.tensor(q_proj.shape[-1], dtype=torch.float32))  # Shape: (batch_size, batch_size)
        
        # Compute attention weights using softmax
        #attention_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, batch_size)
        
        # Compute the final output using attention weights
        #output = torch.matmul(attention_weights, v_proj)  # Shape: (batch_size, latent_dim)
        #adding softmax causes saddle point of 18% validation acc
        output = v_proj
        #output = self.W_p(output)
        return output

    def forward(self, x, gnn_x, embed_x, edge_index, edge_weight=None):
        # Process with TransConv
        x_atten = self.get_attention_feature(gnn_x,embed_x)
        x1 = self.trans_conv_forward(gnn_x, edge_index, edge_weight)
        x1 = self.W_r(x1)
        
        if self.use_graph:
            # Process with GNN if available
            x2 = self.gnn_forward(x_atten, edge_index, edge_weight)
            
            # Aggregate results based on the specified aggregation method
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            elif self.aggregate == 'cat':
                x = torch.cat((x1, x2), dim=1)
            else:
                raise ValueError(f"Invalid aggregate type: {self.aggregate}")
        else:
            x = x1

        # Final fully connected layer
        x = self.fc(x)
        return x
    
    def get_attentions(self, x):
        attns=self.trans_conv.get_attentions(x) # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()

    def trans_conv_forward(self, x, edge_index, edge_weight):
        data = {
            'graph': {
                'node_feat': x,
                'edge_index': edge_index,
                'edge_weight': edge_weight
            }
        }
        return self.trans_conv(data)

    def gnn_forward(self, x, edge_index, edge_weight):
        data = {
            'graph': {
                'node_feat': x,
                'edge_index': edge_index,
                'edge_weight': edge_weight
            }
        }
        return self.gnn(data)
    # def gnn_comp_forward(self, x, edge_index, edge_weight):
    #     data = {
    #         'graph': {
    #             'node_feat': x,
    #             'edge_index': complement_graph(edge_index,len(x)),
    #             'edge_weight': edge_weight
    #         }
    #     }
    #     return self.gnn(data)
      
