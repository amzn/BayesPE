import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


SMALL_CONSTANT = 0.0001


class BayesianLoss(nn.Module):
    """
    Class for BPE loss (equation 4 in the paper)
    """

    def __init__(self):
        super(BayesianLoss, self).__init__()

    def forward(self, scales_logits, probs_y, kl_weight=0.1):
        """
        compute the BPE loss.

        :param scales_logits: 1D array with logits corresponding to the weights for BPE (correspond to log(w) in the paper) [n_ensemble]
        :param probs_y: 2D torch tensor of probabilities assigned to the ground-truth labels over the validation set for each instruction prompt p(y=gt_label|x,a) [n_validation_examples, n_ensemble]
        :param kl_weight: float with weight to assign to entropy term in the BPE cost function. Higher kl_weights encourages more entropy, lower kl_weight encourages more validation likelihood

        :return: BPE cost
        """
        scales = torch.divide(torch.exp(scales_logits), torch.sum(torch.exp(scales_logits))+SMALL_CONSTANT)
        likelihood = torch.sum(torch.log(torch.matmul(probs_y.double(), scales.double())+SMALL_CONSTANT))
        entropy = - torch.dot(scales.double(), torch.exp(scales).double())
        return -likelihood - kl_weight*entropy

class EnsembleScaler(object):
  """
  Class for BPE weights optimiser
  """

  def __init__(self, n_iterations):
      self.n_iterations = n_iterations
      self.criterion = BayesianLoss()


  def scale_ensemble_torch(self, probs, weights):
      """
      scale the probabilities given by each ensemble member according to the given scales (pytorch version).

      :param probs: 3D torch tensor of probabilities corresponding to each ensemble member [n_examples, n_classes, n_ensemble]
      :param weights: 1D torch tensor of weights (w in the paper) [n_ensemble]

      :return: output distributions for BPE [n_examples, n_classes]
      """

      scales = weights.unsqueeze(0)
      scales = scales.unsqueeze(1)
      scaled_probs = scales * probs

      return torch.sum(scaled_probs, axis=2)

  def scale_ensemble(self, probs, weights, n_samp=None):
      """
      scale the probabilities given by each ensemble member according to the given scales.

      :param probs: 3D array of probabilities corresponding to each ensemble member [n_examples, n_classes, n_ensemble]
      :param weights: 1D array of weights (w in the paper) [n_ensemble]
      :param n_samp: number of times to run the LLM, if less then the total amount of probabilities provided

      :return: output distributions for BPE [n_examples, n_classes]
      """
      if n_samp is None:
          n_samp = len(weights)
      chosen_indices = np.argsort(weights)[-n_samp:]
      weights_new = np.zeros(n_samp)
      probs_new = np.zeros((np.shape(probs)[0], np.shape(probs)[1], n_samp))
      for i in range(n_samp):
          weights_new[i] = weights[chosen_indices[i]]
          probs_new[:,:,i] = probs[:,:,chosen_indices[i]]
      return self.scale_ensemble_torch(torch.from_numpy(probs_new), torch.from_numpy(weights_new)).numpy()

  @staticmethod
  def probs_y(probs_pred, gt_labels):
      """
      extract the probabilities associated to the ground-truth labels from probability distributions.

      :param probs_pred: 3D arrays of probabilities corresponding to each ensemble member [n_examples, n_classes, n_ensemble]
      :param gt_labels: 1D array of ground-truth labels (w in the paper) [n_examples]

      :return: 2D array of probabilities assigned to the ground-truth for each prompt instruction [n_examples, n_ensemble]
      """
      probs = np.zeros((np.shape(probs_pred)[0], np.shape(probs_pred)[2]))
      for i in range(np.shape(probs_pred)[0]):
          for j in range(np.shape(probs_pred)[2]):
            probs[i, j] = probs_pred[i, gt_labels[i], j] + SMALL_CONSTANT
      return probs

  def train(self, probs_train, gt_labels, lr=0.001):
      """
      train the weights for BPE.

      :param probs_train: 3D arrays of probabilities corresponding to each ensemble member [n_examples, n_classes, n_ensemble]
      :param gt_labels: 1D array of ground-truth labels (w in the paper) [n_examples]
      :param lr: learning rate for optimiser

      :return: trained weights (w^* in paper) and 1D array of cost values as a function of iterations
      """
      probs_y_train = torch.from_numpy(self.probs_y(probs_train, gt_labels)).cuda()
      scales = nn.Parameter(torch.ones(np.shape(probs_train)[2]).cuda())
      optimizer = optim.LBFGS([scales], lr=lr, max_iter=100, line_search_fn='strong_wolfe')

      scales_values = []
      losses = []

      def _eval():
          loss = self.criterion(scales, probs_y_train)

          loss.backward()
          scales_values.append(scales)
          losses.append(loss.detach().cpu().numpy())
          return loss

      for i in range(self.n_iterations):
          optimizer.step(_eval)
          if i%10 == 0:
            print('iteration {}, loss: {}'.format(i, losses[-1]))

      p_unnorm = np.exp(scales_values[-1].detach().cpu().numpy())
      weights = np.divide(p_unnorm, np.sum(p_unnorm))
      return weights, losses


