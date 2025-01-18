import numpy as np
import torch
import torch.nn as nn

import transformers
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.activations import ACT2FN


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            mask = self.bias[:, :, ns - nd: ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            assert hasattr(
                self, "q_attn"
            ), "If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class AdapterMLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)



class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

        self.adapter_ln = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.adapter_mlp = AdapterMLP(512, config)  # ADAPTER

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))

        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        hidden_states = hidden_states + self.adapter_ln(self.adapter_mlp(hidden_states))

        outputs = [hidden_states] + outputs
        return outputs  # hidden_states, present, (attentions, cross_attentions)




class AdapterDecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            stochastic_policy=False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.stochastic_policy = stochastic_policy
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.transformer.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])


        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        if self.stochastic_policy:
            self.predict_action_mu = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
            self.predict_action_std = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)])
            )
        else:
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        if self.stochastic_policy:
            mu = self.predict_action_mu(x[:,1])[:, -seq_length:, :]
            std = self.predict_action_std(x[:,1])[:, -seq_length:, :]
            std = torch.exp(std)
            eps = torch.distributions.Normal(0, 1).sample()
            action_preds = mu + eps*std
        else:
            action_preds = self.predict_action(x[:,1])[:, -seq_length:, :]  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]


class SoftPromptDecisionTransformer(TrajectoryModel):

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            stochastic_policy=False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

        self.hidden_size = hidden_size
        self.stochastic_policy = stochastic_policy
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # random initialization soft prompt
        self.n_ctx = kwargs["prompt_length"]
        self.deep = kwargs["deep_tuning"]
        self.n_layer = kwargs["n_layer"]
        self.prompt_dropout = nn.Dropout(kwargs["prompt_pdrop"])
        # timestep
        ctx_vectors_timestep = torch.empty(self.n_ctx, 1, dtype=torch.float32)
        nn.init.normal_(ctx_vectors_timestep, std=0.02)
        shape_tup = tuple(ctx_vectors_timestep.shape)
        self.prompt_ctx_timestep = nn.Parameter(ctx_vectors_timestep.view(shape_tup)).requires_grad_(True)  # to be optimized
        # return
        ctx_vectors_return = torch.empty(self.n_ctx, 1, dtype=torch.float32)
        nn.init.normal_(ctx_vectors_return, std=0.02)
        shape_tup = tuple(ctx_vectors_return.shape)
        self.prompt_ctx_return = nn.Parameter(ctx_vectors_return.view(shape_tup)).requires_grad_(True)  # to be optimized
        # state
        ctx_vectors_state = torch.empty(self.n_ctx, self.state_dim, dtype=torch.float32)
        nn.init.normal_(ctx_vectors_state, std=0.02)
        shape_tup = tuple(ctx_vectors_state.shape)
        self.prompt_ctx_state = nn.Parameter(ctx_vectors_state.view(shape_tup)).requires_grad_(True)  # to be optimized
        # action
        ctx_vectors_action = torch.empty(self.n_ctx, self.act_dim, dtype=torch.float32)
        nn.init.normal_(ctx_vectors_action, std=0.02)
        shape_tup = tuple(ctx_vectors_action.shape)
        self.prompt_ctx_action = nn.Parameter(ctx_vectors_action.view(shape_tup)).requires_grad_(True)  # to be optimized
        # attention mask
        self.prompt_ctx_mask = torch.ones(self.n_ctx, dtype=torch.long)

        if self.deep:
            total_d_layer = 3-1
            # timestep
            ctx_vectors_timestep = torch.empty(self.n_ctx, 1, dtype=torch.float32)
            nn.init.normal_(ctx_vectors_timestep, std=0.02)
            shape_tup = tuple(ctx_vectors_timestep.shape)
            self.deep_prompt_ctx_timestep = nn.Parameter(ctx_vectors_timestep.view(shape_tup)).requires_grad_(True)  # to be optimized
            # return
            ctx_vectors_return = torch.empty(self.n_ctx, 1, dtype=torch.float32)
            nn.init.normal_(ctx_vectors_return, std=0.02)
            shape_tup = tuple(ctx_vectors_return.shape)
            self.deep_prompt_ctx_return = nn.Parameter(ctx_vectors_return.view(shape_tup)).requires_grad_(True)  # to be optimized
            # state
            ctx_vectors_state = torch.empty(self.n_ctx, self.state_dim, dtype=torch.float32)
            nn.init.normal_(ctx_vectors_state, std=0.02)
            shape_tup = tuple(ctx_vectors_state.shape)
            self.deep_prompt_ctx_state = nn.Parameter(ctx_vectors_state.view(shape_tup)).requires_grad_(True)  # to be optimized
            # action
            ctx_vectors_action = torch.empty(self.n_ctx, self.act_dim, dtype=torch.float32)
            nn.init.normal_(ctx_vectors_action, std=0.02)
            shape_tup = tuple(ctx_vectors_action.shape)
            self.deep_prompt_ctx_action = nn.Parameter(ctx_vectors_action.view(shape_tup)).requires_grad_(True)  # to be optimized
            # attention mask
            self.deep_prompt_ctx_mask = torch.ones(self.n_ctx, dtype=torch.long)

            self.deep_prompt_embed_timestep = nn.Linear(1, hidden_size)
            self.deep_prompt_embed_return = nn.Linear(1, hidden_size)
            self.deep_prompt_embed_state = nn.Linear(self.state_dim, hidden_size)
            self.deep_prompt_embed_action = nn.Linear(self.act_dim, hidden_size)

        self.prompt_embed_timestep = nn.Linear(1, hidden_size)
        self.prompt_embed_return = nn.Linear(1, hidden_size)
        self.prompt_embed_state = nn.Linear(self.state_dim, hidden_size)
        self.prompt_embed_action = nn.Linear(self.act_dim, hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        if self.stochastic_policy:
            self.predict_action_mu = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
            self.predict_action_std = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)])
            )
        else:
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
        self.predict_return = nn.Linear(hidden_size, 1)

    def incorporate_prompt(self, stacked_inputs, stacked_attention_mask, batch_size, seq_length, device):
        # process prompt the same as d-t
        prompt_states = self.prompt_ctx_state.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        prompt_actions = self.prompt_ctx_action.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        prompt_returns_to_go = self.prompt_ctx_return.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        prompt_timesteps = self.prompt_ctx_timestep.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        prompt_attention_mask = self.prompt_ctx_mask.unsqueeze(0).expand(batch_size, -1).to(device)

        prompt_seq_length = prompt_states.shape[1]
        prompt_state_embeddings = self.prompt_embed_state(prompt_states)
        prompt_action_embeddings = self.prompt_embed_action(prompt_actions)
        if prompt_returns_to_go.shape[1] % 10 == 1:
            prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:,:-1])
        else:
            prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
        prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)

        prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
        prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
        prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings

        prompt_stacked_inputs = torch.stack(
            (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        prompt_stacked_attention_mask = torch.stack(
            (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
        ).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)

        # stacked_inputs add prompted sequence
        if prompt_stacked_inputs.shape[1] == 3 * seq_length: # if only smaple one prompt
            prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
            prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
            stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
            stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
        else: # if sample one prompt for each traj in batch
            stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
            stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)
        
        return stacked_inputs, stacked_attention_mask

    def incorporate_deep_prompt(self, stacked_inputs, stacked_attention_mask, batch_size, seq_length, device):
        # process prompt the same as d-t
        prompt_states = self.deep_prompt_ctx_state.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        prompt_actions = self.deep_prompt_ctx_action.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        prompt_returns_to_go = self.deep_prompt_ctx_return.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        prompt_timesteps = self.deep_prompt_ctx_timestep.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        prompt_attention_mask = self.deep_prompt_ctx_mask.unsqueeze(0).expand(batch_size, -1).to(device)

        prompt_seq_length = prompt_states.shape[1]
        prompt_state_embeddings = self.prompt_embed_state(prompt_states)
        prompt_action_embeddings = self.prompt_embed_action(prompt_actions)
        if prompt_returns_to_go.shape[1] % 10 == 1:
            prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:,:-1])
        else:
            prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
        prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)

        prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
        prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
        prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings

        prompt_stacked_inputs = torch.stack(
            (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        prompt_stacked_attention_mask = torch.stack(
            (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
        ).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)

        # stacked_inputs add prompted sequence
        if prompt_stacked_inputs.shape[1] == 3 * seq_length: # if only smaple one prompt
            prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
            prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
            stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
            stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
        else: # if sample one prompt for each traj in batch
            stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
            prompt_stacked_attention_mask = prompt_stacked_attention_mask.unsqueeze(1)
            prompt_stacked_attention_mask = prompt_stacked_attention_mask.unsqueeze(1)
            stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=-1)
        
        return stacked_inputs, stacked_attention_mask

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        stacked_inputs, stacked_attention_mask = self.incorporate_prompt(
            stacked_inputs, stacked_attention_mask, batch_size, seq_length, states.device
        )

        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        if self.deep:
            transformer_outputs = self.forward_deep_prompt(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
                batch_size=batch_size,
                seq_length=seq_length,
                device=states.device,
            )
        else:
            transformer_outputs = self.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
            )

        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)
        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        if self.stochastic_policy:
            mu = self.predict_action_mu(x[:,1])[:, -seq_length:, :]
            std = self.predict_action_std(x[:,1])[:, -seq_length:, :]
            std = torch.exp(std)
            eps = torch.distributions.Normal(0, 1).sample()
            action_preds = mu + eps*std
        else:
            action_preds = self.predict_action(x[:,1])[:, -seq_length:, :]  # predict next action given state
            
        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        # Note: prompt within kwargs
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]


    def forward_deep_prompt(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            batch_size=None,
            seq_length=None,
            device=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.transformer.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.transformer.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.transformer.config.use_cache
        return_dict = return_dict if return_dict is not None else self.transformer.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.transformer.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.transformer.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.transformer.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.transformer.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.transformer.get_head_mask(head_mask, self.transformer.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        # position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds  # + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.transformer.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.transformer.drop(hidden_states)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.transformer.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.transformer.h, past_key_values)):

            if self.transformer.use_layers is not None and i >= self.transformer.use_layers:
                break

            # Model parallel
            if self.transformer.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = layer_past.to(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.transformer.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                hidden_states, attention_mask = self.incorporate_deep_prompt(
                    hidden_states, attention_mask, batch_size, seq_length, device
                )
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if self.transformer.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.transformer.model_parallel:
                for k, v in self.transformer.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.transformer.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.transformer.ln_f(hidden_states)
        #hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )