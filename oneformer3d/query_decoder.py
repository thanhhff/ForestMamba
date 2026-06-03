import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why BaseModule doesn't call it without us?
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out

@MODELS.register_module()
class QueryDecoder(BaseModule):
    """Query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.num_queries = num_instance_queries + num_semantic_queries
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i]) # [403, 256]
            cls_preds.append(self.out_cls(norm_query))   # [403, 4]
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None    #[403, 1]
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])  #mask_feats:[38584, 256] -> [403, 38584]
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks  #[403, 4]  [403, 1]  [403, 38584]  [403, 38584]

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]  #[38584, 256]  [37025,256]
        mask_feats = [self.x_mask(y) for y in x]   #[38584, 256]  [37025,256]
        queries = self._get_queries(queries, len(x))  #2 x [403, 256]
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)  #2 x [403, 256]
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)


@MODELS.register_module()
class ForAINetv2QueryDecoder(BaseModule):
    """Query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.num_queries = num_instance_queries + num_semantic_queries
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        #self.out_cls = nn.Sequential(
        #    nn.Linear(d_model, d_model), nn.ReLU(),
        #    nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        #cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        pred_scores, pred_masks, attn_masks = [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i]) # [403, 256]
            #cls_preds.append(self.out_cls(norm_query))   # [403, 4]
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None    #[403, 1]
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])  #mask_feats:[38584, 256] -> [403, 38584]
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        #return cls_preds, pred_scores, pred_masks, attn_masks  #[403, 4]  [403, 1]  [403, 38584]  [403, 38584]
        return pred_scores, pred_masks, attn_masks  #[403, 1]  [403, 38584]  [403, 38584]

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        #cls_preds, pred_scores, pred_masks, _ = self._forward_head(
        pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            #cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        #cls_preds, pred_scores, pred_masks = [], [], []
        pred_scores, pred_masks = [], []
        inst_feats = [self.input_proj(y) for y in x]  #[38584, 256]  [37025,256]
        mask_feats = [self.x_mask(y) for y in x]   #[38584, 256]  [37025,256]
        queries = self._get_queries(queries, len(x))  #2 x [403, 256]
        #cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
        pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        #cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)  #2 x [403, 256]
            #cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            #cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            #{'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            {'masks': masks, 'scores': scores}
            for scores, masks in zip(
                pred_scores[:-1], pred_masks[:-1])]
        return dict(
            #cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)

@MODELS.register_module()
class ForAINetv2QueryDecoder_XAwarequery(BaseModule):
    """Query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        if num_instance_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
            self.num_semantic_queries = num_instance_queries + num_semantic_queries
            self.semantic_queries = nn.Embedding(num_semantic_queries, in_channels)
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        #self.out_cls = nn.Sequential(
        #    nn.Linear(d_model, d_model), nn.ReLU(),
        #    nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []

        for i in range(batch_size):
            if len(queries[i]) != 0:
                device = queries[i].device

        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                semantic_queries = self.semantic_queries.weight
                # concat queries[i] and semantic_queries
                if len(queries[i]) == 0:
                    queries[i] = torch.empty(0, *semantic_queries.shape[1:]).to(device)
                concat_queries = torch.cat((queries[i], semantic_queries.to(queries[i].device)), dim=0)
                result_query.append(self.query_proj(concat_queries))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        #cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        pred_scores, pred_masks, attn_masks = [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i]) # [403, 256]
            #cls_preds.append(self.out_cls(norm_query))   # [403, 4]
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None    #[403, 1]
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])  #mask_feats:[38584, 256] -> [403, 38584]
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        #return cls_preds, pred_scores, pred_masks, attn_masks  #[403, 4]  [403, 1]  [403, 38584]  [403, 38584]
        return pred_scores, pred_masks, attn_masks  #[403, 1]  [403, 38584]  [403, 38584]

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        #cls_preds, pred_scores, pred_masks, _ = self._forward_head(
        pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            #cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        #cls_preds, pred_scores, pred_masks = [], [], []
        pred_scores, pred_masks = [], []
        inst_feats = [self.input_proj(y) for y in x]  #[38584, 256]  [37025,256]
        mask_feats = [self.x_mask(y) for y in x]   #[38584, 256]  [37025,256]
        queries = self._get_queries(queries, len(x))  #2 x [403, 256]
        #queries = queries.to(mask_feats.device)
        #cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
        pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        #cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)  #2 x [403, 256]
            #cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            #cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            #{'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            {'masks': masks, 'scores': scores}
            for scores, masks in zip(
                pred_scores[:-1], pred_masks[:-1])]
        return dict(
            #cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)

def _spatial_sort(xyz: torch.Tensor, slab_thickness: float = 5.0,
                  descending: bool = False) -> torch.Tensor:
    """Sort N points by (z_slab, y, x) — approximation of a Z-order curve.

    Args:
        xyz (Tensor): (N, 3) float XYZ coordinates.
        slab_thickness (float): Z-axis bin width in metres.
        descending (bool): Reverse all keys (complementary spatial ordering).

    Returns:
        Tensor: (N,) long permutation indices.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    slab = (z / slab_thickness).long()
    if descending:
        x, y, slab = -x, -y, -slab
    order = torch.argsort(x,       stable=True)
    order = order[torch.argsort(y   [order], stable=True)]
    order = order[torch.argsort(slab[order], stable=True)]
    return order


class MambaAggregation(nn.Module):
    """Local spatial feature aggregation replacing cross-attention (LaSSM §3.2).

    For each query the k spatially nearest superpoints are looked up and their
    features are aggregated with learned softmax kernel weights:

        k_weights = softmax(W_k @ q + b)          shape (Q, k)
        w         = Σ_k k_weights[q,k] * (q[q] ⊙ v[q,k])  shape (Q, d)
        output    = LayerNorm(dropout(W_o @ w) + query)

    When query/source XYZ positions are unavailable the layer falls back to a
    feature-space nearest-neighbour lookup (dot-product similarity), which
    preserves the kernel-aggregation mechanism without spatial grounding.

    Args:
        d_model (int): Feature dimension.
        k (int): Number of nearest neighbours per query.
        dropout (float): Dropout on the output projection.
    """

    def __init__(self, d_model: int, k: int = 8, dropout: float = 0.0):
        super().__init__()
        self.k = k
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Parameter(torch.empty(k, d_model))  # learned kernels
        self.w_b = nn.Parameter(torch.zeros(k))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w_k)

    def forward(self, query, src_feats, query_pos=None, src_pos=None):
        """Forward pass.

        Args:
            query (Tensor): (Q, d) query features.
            src_feats (Tensor): (N, d) source (superpoint) features.
            query_pos (Tensor | None): (Q, 3) query 3-D positions.
            src_pos (Tensor | None): (N, 3) source 3-D positions.

        Returns:
            Tensor: (Q, d) updated query features.
        """
        shortcut = query
        k = min(self.k, src_feats.shape[0])

        # ── k-NN lookup ──────────────────────────────────────────────────
        if query_pos is not None and src_pos is not None:
            dists = torch.cdist(query_pos.float(), src_pos.float())  # (Q, N)
        else:
            # Feature-space fallback: similarity via dot product
            q_n = F.normalize(self.w_q(query), dim=-1)
            s_n = F.normalize(self.w_v(src_feats), dim=-1)
            dists = -torch.mm(q_n, s_n.T)                           # (Q, N)

        _, idx = dists.topk(k, dim=-1, largest=False)               # (Q, k)

        # ── project + gather ─────────────────────────────────────────────
        q   = self.w_q(query)              # (Q, d)
        v   = self.w_v(src_feats)          # (N, d)
        feat = v[idx]                      # (Q, k, d)
        q_exp = q.unsqueeze(1).expand(-1, k, -1)  # (Q, k, d)

        # ── learned softmax kernel weights ───────────────────────────────
        k_w = F.softmax(F.linear(q, self.w_k[:k], self.w_b[:k]), dim=-1)  # (Q, k)

        # ── weighted aggregation ─────────────────────────────────────────
        w = torch.einsum('qk,qkd->qd', k_w, q_exp * feat)           # (Q, d)

        out = self.dropout(self.w_o(w)) + shortcut
        return self.norm(out)


class SSMSpatialSelfLayer(nn.Module):
    """Spatial SSM replacing self-attention on queries (LaSSM §3.3).

    Faithful to the LaSSM paper's SSM class:

    1. Apply pre-norm to queries.
    2. Sort queries by two complementary spatial orderings (ascending and
       descending slab-order, approximating Hilbert + Hilbert-transposed).
    3. Stack both orderings as a batch: shape (2, Q, d).
    4. Pass through a **single shared Mamba instance** — both orderings use
       the same weights (batch dimension = orderings, identical to the paper).
    5. Unsort each ordering back to the original query positions.
    6. **Average** the two outputs (LaSSM: ``sum(paths) / len(paths)``).
    7. Add shortcut residual and apply final-norm.

    When query positions are unavailable the layer falls back to identity
    and reversed-index orderings (no spatial grounding, same shared-weight
    logic otherwise preserved).

    Args:
        d_model (int): Feature dimension.
        d_state (int): Mamba SSM hidden-state size.
        d_conv (int): Mamba depthwise-conv kernel width.
        expand (int): Mamba inner expansion factor.
        slab_thickness (float): Z-axis bin width for spatial slab ordering.
    """

    def __init__(self, d_model: int, d_state: int = 64,
                 d_conv: int = 4, expand: int = 2,
                 slab_thickness: float = 5.0):
        super().__init__()
        self.slab_thickness = slab_thickness
        self.pre_norm   = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        from mamba_ssm import Mamba
        # ONE shared Mamba — processes all orderings as the batch dimension,
        # matching the paper: self.layers[l](query) where query is (num_ord, Q, d).
        self.mamba = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, queries, query_xyzs=None):
        """Forward pass.

        Args:
            queries (List[Tensor]): len batch_size, each (Q_i, d_model).
            query_xyzs (List[Tensor] | None): len batch_size, each (Q_i, 3).
                When provided, queries are serialised spatially before scanning.

        Returns:
            List[Tensor]: Updated queries, same shapes as input.
        """
        outputs = []
        for i, q in enumerate(queries):
            shortcut = q
            normed   = self.pre_norm(q)          # (Q, d)
            pos      = query_xyzs[i] if query_xyzs is not None else None

            if pos is not None and pos.shape[0] > 1:
                ord_a = _spatial_sort(pos, self.slab_thickness, descending=False)
                ord_b = _spatial_sort(pos, self.slab_thickness, descending=True)
            else:
                n     = q.shape[0]
                ord_a = torch.arange(n, device=q.device)
                ord_b = torch.arange(n - 1, -1, -1, device=q.device)

            inv_a = torch.argsort(ord_a)
            inv_b = torch.argsort(ord_b)

            # Stack both orderings as batch dim → (2, Q, d)
            # ONE shared Mamba scans both orderings simultaneously (same weights)
            stacked = torch.stack([normed[ord_a], normed[ord_b]], dim=0)
            out     = self.mamba(stacked)        # (2, Q, d)

            # Unsort each ordering back to original positions then average
            out_a = out[0][inv_a]                # (Q, d)
            out_b = out[1][inv_b]                # (Q, d)
            q_out = (out_a + out_b) / 2          # average — paper: sum/len

            q_out = self.final_norm(shortcut + q_out)
            outputs.append(q_out)
        return outputs


@MODELS.register_module()
class ForAINetv2SSMQueryDecoder_XAwarequery(BaseModule):
    """SSM query decoder faithful to the LaSSM paper (arXiv 2602.11007).

    Each decoder layer applies three operations in order:

    1. **MambaAggregation** — KNN-based local feature aggregation replacing
       cross-attention.  Uses spatial 3-D KNN when ``src_xyzs`` / ``query_xyzs``
       are supplied; falls back to feature-space KNN otherwise.
    2. **SSMSpatialSelfLayer** — two-ordering spatial SSM replacing self-
       attention.  Sorts queries by complementary slab-order curves and
       averages the Mamba outputs.
    3. **FFN** — standard two-layer feed-forward network.

    Mask prediction: dot-product between normalised queries and mask features
    (identical to the transformer decoder it replaces).

    Args:
        num_layers (int): Number of decoder layers.
        num_instance_queries (int): Fixed instance queries (0 for X-aware).
        num_semantic_queries (int): Semantic queries appended after X-aware ones.
        num_classes (int): Foreground classes (unused, kept for interface parity).
        in_channels (int): Input feature channels from backbone.
        d_model (int): Internal model dimension.
        k (int): KNN neighbours for MambaAggregation.
        d_state (int): Mamba SSM hidden-state size.
        d_conv (int): Mamba depthwise-conv kernel width.
        expand (int): Mamba inner expansion factor.
        slab_thickness (float): Z-axis bin width for spatial serialisation.
        hidden_dim (int): FFN hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): FFN activation ('relu' or 'gelu').
        iter_pred (bool): Iterative prediction (aux losses per layer).
        attn_mask (bool): Use predicted mask to gate MambaAggregation KNN.
        objectness_flag (bool): Predict per-query objectness score.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model,
                 iter_pred, attn_mask, objectness_flag,
                 k=8, d_state=64, d_conv=4, expand=2,
                 slab_thickness=5.0,
                 hidden_dim=1024, dropout=0.0, activation_fn='gelu',
                 **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.iter_pred  = iter_pred
        self.attn_mask  = attn_mask

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())

        if num_instance_queries > 0:
            self.query = nn.Embedding(
                num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
            self.num_queries = num_semantic_queries
            self.semantic_queries = nn.Embedding(num_semantic_queries, in_channels)

        self.agg_layers  = nn.ModuleList()   # MambaAggregation (≈ cross-attn)
        self.ssm_layers  = nn.ModuleList()   # SSMSpatialSelfLayer (≈ self-attn)
        self.ffn_layers  = nn.ModuleList()
        for _ in range(num_layers):
            self.agg_layers.append(MambaAggregation(d_model, k, dropout))
            self.ssm_layers.append(
                SSMSpatialSelfLayer(d_model, d_state, d_conv, expand, slab_thickness))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))

        self.out_norm = nn.LayerNorm(d_model)
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))

    # ------------------------------------------------------------------
    # helpers (identical interface to ForAINetv2QueryDecoder_XAwarequery)
    # ------------------------------------------------------------------

    def _get_queries(self, queries=None, batch_size=None):
        if batch_size is None:
            batch_size = len(queries)

        device = None
        for i in range(batch_size):
            if len(queries[i]) != 0:
                device = queries[i].device
                break

        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                sem_q = self.semantic_queries.weight
                q_i   = queries[i]
                if len(q_i) == 0:
                    q_i = torch.empty(0, sem_q.shape[-1]).to(device)
                concat = torch.cat((q_i, sem_q.to(q_i.device)), dim=0)
                result_query.append(self.query_proj(concat))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        pred_scores, pred_masks, attn_masks = [], [], []
        for i in range(len(queries)):
            norm_q = self.out_norm(queries[i])
            pred_scores.append(
                self.out_score(norm_q) if self.objectness_flag else None)
            pred_mask = torch.einsum('nd,md->nm', norm_q, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[attn_mask.sum(-1) == attn_mask.shape[-1]] = False
                attn_masks.append(attn_mask.detach())
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return pred_scores, pred_masks, attn_masks

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def _layer_forward(self, queries, inst_feats, query_xyzs, src_xyzs, layer_idx):
        """Run one decoder layer (Aggregation → SSM → FFN) over the batch."""
        agg = self.agg_layers[layer_idx]
        ssm = self.ssm_layers[layer_idx]
        ffn = self.ffn_layers[layer_idx]

        # _get_queries appends num_semantic_queries class-level tokens after
        # the instance seeds, so queries[i] may be longer than query_xyzs[i].
        # Pad the position list with the scene centroid for those extra tokens.
        padded_xyzs = None
        if query_xyzs is not None:
            padded_xyzs = []
            for i, q in enumerate(queries):
                q_pos = query_xyzs[i]
                if q_pos is not None and q_pos.shape[0] < q.shape[0]:
                    n_extra = q.shape[0] - q_pos.shape[0]
                    s_pos   = src_xyzs[i] if src_xyzs is not None else None
                    pad = (s_pos.mean(0, keepdim=True).expand(n_extra, -1)
                           if s_pos is not None
                           else torch.zeros(n_extra, 3,
                                            device=q_pos.device,
                                            dtype=q_pos.dtype))
                    q_pos = torch.cat([q_pos, pad], dim=0)
                padded_xyzs.append(q_pos)

        new_queries = []
        for i, (q, src) in enumerate(zip(queries, inst_feats)):
            q_pos = padded_xyzs[i] if padded_xyzs is not None else None
            s_pos = src_xyzs[i]    if src_xyzs   is not None else None
            q = agg(q, src, query_pos=q_pos, src_pos=s_pos)
            new_queries.append(q)

        new_queries = ssm(new_queries, padded_xyzs)
        new_queries = ffn(new_queries)
        return new_queries

    def forward_simple(self, x, queries, query_xyzs=None, src_xyzs=None):
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries    = self._get_queries(queries, len(x))
        for i in range(len(self.agg_layers)):
            queries = self._layer_forward(queries, inst_feats, query_xyzs, src_xyzs, i)
        pred_scores, pred_masks, _ = self._forward_head(queries, mask_feats)
        return dict(masks=pred_masks, scores=pred_scores)

    def forward_iter_pred(self, x, queries, query_xyzs=None, src_xyzs=None):
        pred_scores, pred_masks = [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries    = self._get_queries(queries, len(x))

        pred_score, pred_mask, _ = self._forward_head(queries, mask_feats)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)

        for i in range(len(self.agg_layers)):
            queries = self._layer_forward(queries, inst_feats, query_xyzs, src_xyzs, i)
            pred_score, pred_mask, _ = self._forward_head(queries, mask_feats)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'masks': m, 'scores': s}
            for s, m in zip(pred_scores[:-1], pred_masks[:-1])]
        return dict(
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None, query_xyzs=None, src_xyzs=None):
        """Forward pass.

        Args:
            x (List[Tensor]): Backbone features, each (N_i, in_channels).
            queries (List[Tensor]): X-aware query features, each (Q_i, in_channels).
            query_xyzs (List[Tensor] | None): Query 3-D positions, each (Q_i, 3).
                Enables spatial KNN in MambaAggregation and spatial serialisation
                in SSMSpatialSelfLayer.  When None both layers use feature-space
                fallbacks.
            src_xyzs (List[Tensor] | None): Source superpoint positions, each (N_i, 3).
                Required for spatial KNN in MambaAggregation alongside query_xyzs.

        Returns:
            Dict with keys ``masks``, ``scores``, and optionally ``aux_outputs``.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries, query_xyzs, src_xyzs)
        return self.forward_simple(x, queries, query_xyzs, src_xyzs)


@MODELS.register_module()
class ScanNetQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, **kwargs)
        assert num_semantic_linears in [1, 2]
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = \
            [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
            self._forward_head(queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
                self._forward_head(queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                sem_preds=sem_pred,
                masks=masks,
                scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1],
                pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)


@MODELS.register_module()
class OneDataQueryDecoder(BaseModule):
    """Query decoder. The same as above, but for 2 datasets.

    Args:
        num_layers (int): Number of transformer layers.
        num_queries_1dataset (int): Number of queries for the first dataset.
        num_queries_2dataset (int): Number of queries for the second dataset.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        prefix_2dataset (string): Prefix for the second dataset.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, 
                 num_layers, 
                 num_queries_1dataset, 
                 num_queries_2dataset,
                 num_classes_1dataset, 
                 num_classes_2dataset,
                 prefix_1dataset,
                 prefix_2dataset,
                 in_channels, 
                 d_model, 
                 num_heads, 
                 hidden_dim,
                 dropout, 
                 activation_fn, 
                 iter_pred, 
                 attn_mask, 
                 fix_attention, 
                 **kwargs):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())

        self.num_queries_1dataset = num_queries_1dataset
        self.num_queries_2dataset = num_queries_2dataset

        self.queries_1dataset = nn.Embedding(num_queries_1dataset, d_model)
        self.queries_2dataset = nn.Embedding(num_queries_2dataset, d_model)
        
        self.prefix_1dataset = prefix_1dataset 
        self.prefix_2dataset = prefix_2dataset

        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls_1dataset = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes_1dataset + 1))
        self.out_cls_2dataset = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes_2dataset + 1))
        self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.num_classes_1dataset = num_classes_1dataset 
        self.num_classes_2dataset = num_classes_2dataset

    def _get_queries(self, batch_size, scene_names):
        """Get query tensor.

        Args:
            batch_size (int, optional): batch size.
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        
        result_queries = []
        for i in range(batch_size):
            if self.prefix_1dataset in scene_names[i]:
                result_queries.append(self.queries_1dataset.weight)
            elif self.prefix_2dataset in scene_names[i]:
                result_queries.append(self.queries_2dataset.weight)
            else:
                raise RuntimeError(f'Invalid scene name "{scene_names[i]}".')

        return result_queries

    def _forward_head(self, queries, mask_feats, scene_names):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            scene_names (List[string]): list of len batch size, which 
                contains scene names.

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor]: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            
            if self.prefix_1dataset in scene_names[i]:
                cls_preds.append(self.out_cls_1dataset(norm_query))
            elif self.prefix_2dataset in scene_names[i]:
                cls_preds.append(self.out_cls_2dataset(norm_query))
            else:
                raise RuntimeError(f'Invalid scene name "{scene_names[i]}".')
            

            pred_scores.append(self.out_score(norm_query))
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        return  cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, scene_names):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(len(x), scene_names)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, scene_names)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, scene_names):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(len(x), scene_names)
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats, scene_names)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats, scene_names)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, scene_names):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, scene_names)
        else:
            return self.forward_simple(x, scene_names)
