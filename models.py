import torch

from transformers import AutoModel
from torch import nn

class GRLayer(torch.autograd.Function):
    @staticmethod
    def forward(bkw, x, lambda_value=0.01):
        bkw.lambda_value = torch.tensor(lambda_value)
        return x.reshape_as(x)

    @staticmethod
    def backward(bkw, prev_gradient):
        post_gradient = prev_gradient.clone()
        return bkw.lambda_value * post_gradient.neg(), None


class AdversarialNet(torch.nn.Module):
  def __init__(self, config):
    super(AdversarialNet, self).__init__()

    self.num_classes = config["num_classes"]
    self.num_classes_adv = config["num_classes_adv"]
    self.embed_size = config["embed_size"]
    self.first_last_avg = config["first_last_avg"]

    self.plm = AutoModel.from_pretrained(config["model_name"])
    config = self.plm.config
    config.type_vocab_size = 4
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings)

    for param in self.plm.parameters():
      param.requires_grad = True

    self.linear_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes)
    self.linear_layer_adv = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes_adv)
    self.task_linear = torch.nn.Linear(in_features=self.embed_size, out_features=2)

    self.multi_head_att = torch.nn.MultiheadAttention(self.embed_size, 8, batch_first=True)
    self.Q = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.V = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    self._init_weights(self.linear_layer)
    self._init_weights(self.linear_layer_adv)
    self._init_weights(self.Q)
    self._init_weights(self.K)
    self._init_weights(self.V)
    self._init_weights(self.multi_head_att)
    self._init_weights(self.task_linear)

  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1, position_sep, visualize=False):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)

    last_sent1, first_sent1 = out_sent1.hidden_states[-1], out_sent1.hidden_states[1]

    if self.first_last_avg:
      embed_sent1 = torch.div((last_sent1 + first_sent1), 2)
    else:
      embed_sent1 = last_sent1

    tar_mask_sent1 = (segs_sent1 == 0).long()
    tar_mask_sent2 = (segs_sent1 == 1).long()

    H_sent1 = torch.mul(tar_mask_sent1.unsqueeze(2), embed_sent1)
    H_sent2 = torch.mul(tar_mask_sent2.unsqueeze(2), embed_sent1)

    K_sent1 = self.K(H_sent1)
    V_sent1 = self.V(H_sent1)
    Q_sent2 = self.Q(H_sent2)

    att_output = self.multi_head_att(Q_sent2, K_sent1, V_sent1)

    H_sent = torch.mean(att_output[0], dim=1)

    if visualize:
      return H_sent
    
    if self.training:
      batch_size = H_sent.shape[0]
      samples = H_sent[:batch_size // 2, :]
      samples_adv = H_sent[batch_size // 2:, ]

      predictions = self.linear_layer(samples)
      predictions_adv = self.linear_layer_adv(samples_adv)

      mean_grl = GRLayer.apply(torch.mean(embed_sent1, dim=1), .01)
      task_prediction = self.task_linear(mean_grl)

      return predictions, predictions_adv, task_prediction
    else:
      predictions = self.linear_layer(H_sent)

      return predictions


class BaselineModel(torch.nn.Module):
  def __init__(self, config):
    super(BaselineModel, self).__init__()

    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]
    self.first_last_avg = config["first_last_avg"]

    self.plm = AutoModel.from_pretrained(config["model_name"])
    config = self.plm.config
    config.type_vocab_size = 4
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings)

    for param in self.plm.parameters():
      param.requires_grad = True

    self.linear_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes)
    self.multi_head_att = torch.nn.MultiheadAttention(self.embed_size, 8, batch_first=True)
    self.Q = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.V = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    self._init_weights(self.linear_layer)
    self._init_weights(self.Q)
    self._init_weights(self.K)
    self._init_weights(self.V)
    self._init_weights(self.multi_head_att)

  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1, position_sep, visualize=False):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)

    last_sent1, first_sent1 = out_sent1.hidden_states[-1], out_sent1.hidden_states[1]

    if self.first_last_avg:
      embed_sent1 = torch.div((last_sent1 + first_sent1), 2)
    else:
      embed_sent1 = last_sent1

    tar_mask_sent1 = (segs_sent1 == 0).long()
    tar_mask_sent2 = (segs_sent1 == 1).long()

    H_sent1 = torch.mul(tar_mask_sent1.unsqueeze(2), embed_sent1)
    H_sent2 = torch.mul(tar_mask_sent2.unsqueeze(2), embed_sent1)

    K_sent1 = self.K(H_sent1)
    V_sent1 = self.V(H_sent1)
    Q_sent2 = self.Q(H_sent2)

    att_output = self.multi_head_att(Q_sent2, K_sent1, V_sent1)

    H_sent = torch.mean(att_output[0], dim=1)

    if visualize:
      return H_sent

    predictions = self.linear_layer(H_sent)

    return predictions
