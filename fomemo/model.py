"""Neural modules for FoMEMO training and inference."""

from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.utils.checkpoint import checkpoint

from .config import MODEL_CONFIG


def _attention_mask(mask: Tensor) -> Tensor:
    return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class VariableNumFeaturesEncoder(nn.Module):
    def __init__(self, base_encoder, num_features):
        super().__init__()
        self.base_encoder = base_encoder
        self.num_features = num_features

    def forward(self, x):
        x = x * (self.num_features / x.shape[-1])
        padding = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        return self.base_encoder(torch.cat((x, padding), dim=-1))


class Linear(nn.Linear):
    def __init__(self, num_features, embedding_size, replace_nan_by_zero=False):
        super().__init__(num_features, embedding_size)
        self.num_features = num_features
        self.embedding_size = embedding_size
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return super().forward(x)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("replace_nan_by_zero", True)


class NoPositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len=None):
        super().__init__()

    def forward(self, x):
        return x


class FoMEMOTransformerEncoderLayer(Module):
    """Transformer encoder layer used by the released inference model."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        pre_norm=False,
        device=None,
        dtype=None,
        recompute_attn=False,
        save_trainingset_representations=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.save_trainingset_representations = save_trainingset_representations
        self.saved_src_to_attend_to = None
        self.activation = torch.nn.functional.relu if activation == "relu" else torch.nn.functional.gelu

    def __setstate__(self, state):
        state.setdefault("activation", torch.nn.functional.relu)
        super().__setstate__(state)
        self.__dict__.setdefault("save_trainingset_representations", False)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.save_trainingset_representations:
            assert isinstance(src_mask, int) and not self.training

        src_normalized = self.norm1(src) if self.pre_norm else src
        if isinstance(src_mask, tuple):
            assert not self.self_attn.batch_first
            assert src_key_padding_mask is None
            global_mask, train_mask, eval_mask = src_mask
            num_global = global_mask.shape[0]
            num_train = train_mask.shape[0]
            global_tokens = src_normalized[:num_global]
            train_tokens = src_normalized[num_global : num_global + num_train]
            global_and_train = src_normalized[: num_global + num_train]
            eval_tokens = src_normalized[num_global + num_train :]
            attention = (
                partial(checkpoint, self.self_attn)
                if self.recompute_attn
                else self.self_attn
            )
            global_output = attention(
                global_tokens,
                global_and_train,
                global_and_train,
                None,
                True,
                global_mask,
            )[0]
            train_output = attention(
                train_tokens,
                global_tokens,
                global_tokens,
                None,
                True,
                train_mask,
            )[0]
            eval_output = attention(
                eval_tokens,
                src_normalized,
                src_normalized,
                None,
                True,
                eval_mask,
            )[0]
            attention_output = torch.cat(
                [global_output, train_output, eval_output],
                dim=0,
            )
        elif isinstance(src_mask, int):
            assert src_key_padding_mask is None
            split = src_mask
            attended_training = src_normalized[:split]
            if self.save_trainingset_representations:
                if split == src_normalized.shape[0] or split is None:
                    self.saved_src_to_attend_to = attended_training
                elif split == 0:
                    if self.saved_src_to_attend_to is None:
                        raise ValueError("Training representations have not been saved.")
                    attended_training = self.saved_src_to_attend_to
                else:
                    raise ValueError("Unsupported representation cache position.")
            left = self.self_attn(
                src_normalized[:split],
                src_normalized[:split],
                src_normalized[:split],
            )[0]
            right = self.self_attn(
                src_normalized[split:],
                attended_training,
                attended_training,
            )[0]
            attention_output = torch.cat([left, right], dim=0)
        elif self.recompute_attn:
            attention_output = checkpoint(
                self.self_attn,
                src_normalized,
                src_normalized,
                src_normalized,
                src_key_padding_mask,
                True,
                src_mask,
            )[0]
        else:
            attention_output = self.self_attn(
                src_normalized,
                src_normalized,
                src_normalized,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )[0]

        src = src + self.dropout1(attention_output)
        if not self.pre_norm:
            src = self.norm1(src)
        src_normalized = self.norm2(src) if self.pre_norm else src
        feedforward = self.linear2(
            self.dropout(self.activation(self.linear1(src_normalized)))
        )
        src = src + self.dropout2(feedforward)
        return src if self.pre_norm else self.norm2(src)


class FoMEMOTransformerEncoder(Module):
    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [encoder_layer_creator() for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        return self.norm(output) if self.norm is not None else output


class FoMEMOTransformer(nn.Module):
    """Preference-conditioned Transformer for multi-objective optimization."""

    def __init__(self, encoder, y_encoder, style_encoder, embedding_size=None,
                 num_heads=None, hidden_size=None, num_layers=None, num_outputs=None):
        super().__init__()
        embedding_size = MODEL_CONFIG["embedding_size"] if embedding_size is None else embedding_size
        num_heads = MODEL_CONFIG["attention_heads"] if num_heads is None else num_heads
        hidden_size = MODEL_CONFIG["hidden_size"] if hidden_size is None else hidden_size
        num_layers = MODEL_CONFIG["transformer_layers"] if num_layers is None else num_layers
        num_outputs = MODEL_CONFIG["output_logits_size"] if num_outputs is None else num_outputs
        self.model_type = "Transformer"
        self.transformer_encoder = FoMEMOTransformerEncoder(
            lambda: FoMEMOTransformerEncoderLayer(
                embedding_size, num_heads, hidden_size, dropout=0.0, activation="gelu"
            ), num_layers,
        )
        self.ninp = embedding_size
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = NoPositionalEncoding(embedding_size)
        self.return_all_outputs = False
        self.decoder_dict = nn.ModuleDict({"standard": nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, num_outputs)
        )})
        self.decoder_dict_once = None
        self.decoder_dict_once_embeddings = None
        self.input_ln = None
        self.style_encoder = style_encoder
        self.init_method = None
        self.global_att_embeddings = None
        self.full_attention = False
        self.efficient_eval_masking = True
        self.nhid = hidden_size
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("efficient_eval_masking", False)
        self.__dict__.setdefault("decoder_dict_once", None)
        self.__dict__.setdefault("return_all_outputs", False)

        def update_gelu(module):
            if isinstance(module, nn.GELU):
                module.__dict__.setdefault("approximate", "none")

        self.apply(update_gelu)

    @staticmethod
    def generate_d_query_matrix(size, query_size):
        train_size = size - query_size
        mask = torch.ones(size, size, dtype=torch.bool)
        mask[:, train_size:].zero_()
        mask |= torch.eye(size, dtype=torch.bool)
        return _attention_mask(mask)

    def forward(self, *args, **kwargs):
        if len(args) == 3:
            allowed = {"src_mask", "style", "only_return_standard_out"}
            unknown = set(kwargs) - allowed
            if unknown:
                raise TypeError(f"Unsupported keyword arguments: {sorted(unknown)}")
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=0)
            style = kwargs.pop("style", None)
            return self._forward(
                (style, x, args[1]),
                single_eval_pos=len(args[0]),
                **kwargs,
            )
        if len(args) == 1 and isinstance(args[0], tuple):
            return self._forward(args[0], **kwargs)
        raise TypeError("Expected (train_x, train_y, test_x) or one input tuple.")

    def _forward(
        self,
        src,
        src_mask=None,
        single_eval_pos=None,
        only_return_standard_out=True,
    ):
        if not isinstance(src, tuple):
            raise TypeError("Inputs must be provided as a tuple.")
        if len(src) == 2:
            src = (None,) + src
        style_src, x_src, y_src = src
        if single_eval_pos is None:
            single_eval_pos = x_src.shape[0]

        x_src = self.encoder(x_src)
        if self.decoder_dict_once is not None:
            repeated = self.decoder_dict_once_embeddings.repeat(
                1,
                x_src.shape[1],
                1,
            )
            x_src = torch.cat([x_src, repeated], dim=0)
        y_src = self.y_encoder(y_src)

        if self.style_encoder:
            if style_src is None:
                raise ValueError("A preference vector is required.")
            style_src = self.style_encoder(style_src).unsqueeze(0)
        else:
            style_src = torch.empty(0, device=x_src.device)
        global_src = (
            torch.empty(0, device=x_src.device)
            if self.global_att_embeddings is None
            else self.global_att_embeddings.weight.unsqueeze(1).repeat(
                1,
                x_src.shape[1],
                1,
            )
        )

        if src_mask is None:
            if self.global_att_embeddings is not None:
                raise NotImplementedError(
                    "Global attention is not used by the released inference model."
                )
            full_len = len(x_src) + len(style_src)
            if self.full_attention:
                src_mask = _attention_mask(
                    torch.ones((full_len, full_len), dtype=torch.bool)
                ).to(x_src.device)
            elif self.efficient_eval_masking:
                src_mask = single_eval_pos + len(style_src)
            else:
                src_mask = self.generate_d_query_matrix(
                    full_len,
                    len(x_src) - single_eval_pos,
                ).to(x_src.device)

        training_x = x_src[:single_eval_pos]
        if y_src is not None:
            training_x = training_x + y_src[:single_eval_pos]
        encoded = torch.cat(
            [global_src, style_src, training_x, x_src[single_eval_pos:]],
            dim=0,
        )
        if self.input_ln is not None:
            encoded = self.input_ln(encoded)
        if self.pos_encoder is not None:
            encoded = self.pos_encoder(encoded)
        output = self.transformer_encoder(encoded, src_mask)

        prefix = len(style_src)
        if self.global_att_embeddings is not None:
            prefix += self.global_att_embeddings.num_embeddings
        start = prefix if self.return_all_outputs else single_eval_pos + prefix
        end = (
            -len(self.decoder_dict_once_embeddings)
            if self.decoder_dict_once is not None
            else None
        )
        once = (
            {
                key: decoder(output[-(index + 1)])
                for index, (key, decoder) in enumerate(
                    self.decoder_dict_once.items()
                )
            }
            if self.decoder_dict_once is not None
            else {}
        )
        decoded = (
            {
                key: decoder(output[start:end])
                for key, decoder in self.decoder_dict.items()
            }
            if self.decoder_dict is not None
            else {}
        )
        if only_return_standard_out:
            return decoded["standard"]
        return (decoded, once) if once else decoded
