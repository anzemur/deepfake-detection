import torch
import torch.nn as nn
import torch.nn.functional as F

class PairNN(nn.Module):
  def __init__(self, n_classes=1, fc_layers=None):
    super().__init__()

    self.gamma = self._get_gamma()
    output = self._get_fcl_input_size()
    fc_layers = fc_layers if fc_layers is not None else output

    self.fc_face = nn.Linear(output, n_classes)

    self.fc_pair1 = nn.Linear(2*output, fc_layers)
    self.fc_pair2 = nn.Linear(fc_layers, n_classes)

    self.fc_pair = nn.Linear(2*output, n_classes)

  def _get_fcl_input_size(self):
    raise NotImplementedError()

  def _get_gamma(self):
    raise NotImplementedError()

  def _forward(self, X):
    """
    Extending model's forward function. This needs to be implemented
    in the extending model. Funcion gets called in `self.forward()`.

    Args:
      X (tensor): Input tensor.

    Returns:
      Output of the extending model after processing.
    """
    raise NotImplementedError()

  def _pair_forward(self, X_f, X_nf):
    X_f = self._forward(X_f)
    X_nf = self._forward(X_nf)

    X_pair = torch.cat((X_f, X_nf), dim=1)
    X_pair = self.fc_pair(X_pair)

    # Another scaled fc layer.
    # X_pair = F.relu(self.fc_pair1(X_pair))
    # # X_pair = torch.sigmoid(self.fc_pair2(X_pair))
    # X_pair = self.fc_pair2(X_pair)

    # X_f = torch.sigmoid(self.fc_face(X_f))
    X_f = self.fc_face(X_f)
  
    return torch.squeeze(X_f), torch.squeeze(X_pair)

  def forward(self, X):
    if self.training:
      out_face = []

      for X_f, X_nf in X:
        X_f = self._forward(X_f)
        # X_f = torch.sigmoid(self.fc_face(X_f))
        X_f = self.fc_face(X_f)

        out_face.append(torch.squeeze(X_f))

      return torch.sum(torch.stack(out_face)) / len(out_face)
    
    else:
      out_face = []
      out_pair = []

      for X_f, X_nf in X:
        out_f, out_p = self._pair_forward(X_f, X_nf)
        out_face.append(out_f)
        out_pair.append(out_p)

      score_f = torch.sum(torch.stack(out_face)) / len(out_face)
      score_p = torch.sum(torch.stack(out_pair)) / len(out_pair)

      # return score_f +  score_p * self.gamma
      return torch.sigmoid(score_f) +  torch.sigmoid(score_p) * self.gamma

    # out_face = []
    # out_pair = []

    # for X_f, X_nf in X:
    #   out_f, out_p = self._pair_forward(X_f, X_nf)
    #   out_face.append(out_f)
    #   out_pair.append(out_p)

    # score_f = torch.sum(torch.stack(out_face)) / len(out_face)
    # score_p = torch.sum(torch.stack(out_pair)) / len(out_pair)

    # if self.training:
    #   return score_f + score_p * 1
    # else:
    #   return torch.sigmoid(score_f) +  torch.sigmoid(score_p) * self.gamma
