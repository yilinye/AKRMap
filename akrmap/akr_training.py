import datetime
import json
from math import log2
from typing import Optional
import os

import torch

from torch import Tensor, tensor, eye, device, ones, isnan, zeros
from torch import max as torch_max
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from numpy import array
from numpy import save as np_save
from tqdm import tqdm
from torch import nn
import random

from akrmap.utils import get_random_string, entropy, distance_functions
from config import config

EPS = tensor([1e-10]).to(device(config.dev))
# EPS = torch.tensor([1e-10])


def calculate_optimized_p_cond(input_points: tensor,
                               target_entropy: float,
                               dist_func: str,
                               tol: float,
                               max_iter: int,
                               min_allowed_sig_sq: float,
                               max_allowed_sig_sq: float,
                               dev: str) -> Optional[tensor]:
    """
    Calculates conditional probability matrix optimized by binary search
    :param input_points: A matrix of input data where every row is a data point
    :param target_entropy: The entropy that every distribution (row) in conditional
    probability matrix will be optimized to match
    :param dist_func: A name for desirable distance function (e.g. "euc", "jaccard" etc)
    :param tol: A small number - tolerance threshold for binary search
    :param max_iter: Maximum number of binary search iterations
    :param min_allowed_sig_sq: Minimum allowed value for the spread of any distribution
    in conditional probability matrix
    :param max_allowed_sig_sq: Maximum allowed value for the spread of any distribution
    in conditional probability matrix
    :param dev: device for tensors (e.g. "cpu" or "cuda")
    :return:
    """
    n_points = input_points.size(0)

    # Calculating distance matrix with the given distance function
    dist_f = distance_functions[dist_func]
    distances = dist_f(input_points)
    diag_mask = (1 - eye(n_points)).to(device(dev))

    # Initializing sigmas
    min_sigma_sq = (min_allowed_sig_sq + 1e-20) * ones(n_points).to(device(dev))
    max_sigma_sq = max_allowed_sig_sq * ones(n_points).to(device(dev))
    sq_sigmas = (min_sigma_sq + max_sigma_sq) / 2

    # Computing conditional probability matrix from distance matrix
    p_cond = get_p_cond(distances, sq_sigmas, diag_mask)

    # Making a vector of differences between target entropy and entropies for all rows in p_cond
    ent_diff = entropy(p_cond) - target_entropy

    # Binary search ends when all entropies match the target entropy
    finished = ent_diff.abs() < tol

    curr_iter = 0
    while not finished.all().item():
        if curr_iter >= max_iter:
            print("Warning! Exceeded max iter.", flush=True)
            # print("Discarding batch")
            return p_cond
        pos_diff = (ent_diff > 0).float()
        neg_diff = (ent_diff <= 0).float()

        max_sigma_sq = pos_diff * sq_sigmas + neg_diff * max_sigma_sq
        min_sigma_sq = pos_diff * min_sigma_sq + neg_diff * sq_sigmas

        sq_sigmas = finished.logical_not() * (min_sigma_sq + max_sigma_sq) / 2 + finished * sq_sigmas
        p_cond = get_p_cond(distances, sq_sigmas, diag_mask)
        ent_diff = entropy(p_cond) - target_entropy
        finished = ent_diff.abs() < tol
        curr_iter += 1
    if isnan(ent_diff.max()):
        print("Warning! Entropy is nan. Discarding batch", flush=True)
        return
    return p_cond


def get_p_cond(distances: tensor, sigmas_sq: tensor, mask: tensor) -> tensor:
    """
    Calculates conditional probability distribution given distances and squared sigmas
    :param distances: Matrix of squared distances ||x_i - x_j||^2
    :param sigmas_sq: Row vector of squared sigma for each row in distances
    :param mask: A mask tensor to set diagonal elements to zero
    :return: Conditional probability matrix
    """
    logits = -distances / (2 * torch_max(sigmas_sq, EPS).view(-1, 1))
    logits.exp_()
    masked_exp_logits = logits * mask
    normalization = torch_max(masked_exp_logits.sum(1), EPS).unsqueeze(1)
    return masked_exp_logits / normalization + 1e-10


def get_q_joint(emb_points: tensor, dist_func: str, alpha: int, ) -> tensor:
    """
    Calculates the joint probability matrix in embedding space.
    :param emb_points: Points in embeddings space
    :param alpha: Number of degrees of freedom in t-distribution
    :param dist_func: A kay name for a distance function
    :return: Joint distribution matrix in emb. space
    """
    n_points = emb_points.size(0)
    mask = (-eye(n_points) + 1).to(emb_points.device)
    dist_f = distance_functions[dist_func]
    distances = dist_f(emb_points) / alpha
    q_joint = (1 + distances).pow(-(1 + alpha) / 2) * mask
    q_joint /= q_joint.sum()
    return torch_max(q_joint, EPS)


def make_joint(distr_cond: tensor) -> tensor:
    """
    Makes a joint probability distribution out of conditional distribution
    :param distr_cond: Conditional distribution matrix
    :return: Joint distribution matrix. All values in it sum up to 1.
    Too small values are set to fixed epsilon
    """
    n_points = distr_cond.size(0)
    distr_joint = (distr_cond + distr_cond.t()) / (2 * n_points)
    return torch_max(distr_joint, EPS)


def loss_function(p_joint: Tensor, q_joint: Tensor) -> Tensor:
    """
    Calculates KLDiv between joint distributions in original and embedding space
    :param p_joint:
    :param q_joint:
    :return: KLDiv value
    """
    # TODO Add here alpha gradient calculation too
    # TODO Add L2-penalty for early compression?
    return (p_joint * torch.log((p_joint + EPS) / (q_joint + EPS))).sum()

class embedding_score_data(Dataset):
    def __init__(self, input_points, scores):
        self.input_points = input_points
        self.scores = scores
    def __len__(self):
        return len(self.input_points)

    def __getitem__(self, idx):
        return self.input_points[idx], self.scores[idx]
    
def sample_and_get_remaining_ids(N, n):
    # Ensure n is not larger than N
    if n > N:
        raise ValueError("n cannot be larger than N")

    # Sample n unique IDs from the range 0 to N-1
    sampled_ids = random.sample(range(N), n)

    # Convert sampled_ids to a set for faster look-up
    sampled_set = set(sampled_ids)

    # Get the remaining IDs
    remaining_ids = [i for i in range(N) if i not in sampled_set]

    return sampled_ids, remaining_ids
    


class MapLoss(nn.Module):
    def __init__(self):
        super(MapLoss, self).__init__()
        self.a1 = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.a2 = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.b = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, embeddings, scores, p_joint):

        N=embeddings.shape[0]
        n=int(0.9*N)
        train_ids, val_ids=sample_and_get_remaining_ids(N, n)

        X_train=embeddings[train_ids]
        X_val=embeddings[val_ids]
        dists = torch.cdist(X_train, X_train)
        dists_val = torch.cdist(X_train, X_val)
        ra2=self.a2**2
        ra1=self.a1**2
        rb=self.b**2
        kernels=1/(1+ra2*dists**(2*rb))
        kernels_val=1/(1+ra2*dists_val**(2*rb))

        w_kde=scores[train_ids] @ (kernels)
        kde=torch.sum(kernels, axis=0)
        kde_reg=w_kde/kde

        w_kde_val=scores[train_ids] @ (kernels_val)
        kde_val=torch.sum(kernels_val, axis=0)
        kde_reg_val=w_kde_val/kde_val

        train_loss=torch.sum((scores[train_ids]-ra1*kde_reg)**2)/len(train_ids)
        val_loss=torch.sum((scores[val_ids]-ra1*kde_reg_val)**2)/len(val_ids)
         
        loss=0.3*train_loss+val_loss
        return loss
def fit_akrmap_model(model: torch.nn.Module,
                    input_points: tensor,
                    opt: Optimizer,
                    perplexity: Optional[int],
                    n_epochs: int,
                    dev: str,
                    save_dir_path: str,
                    epochs_to_save_after: Optional[int],
                    early_exaggeration: int,
                    early_exaggeration_constant: int,
                    batch_size: int,
                    dist_func_name: str,
                    bin_search_tol: float,
                    bin_search_max_iter: int,
                    min_allowed_sig_sq: float,
                    max_allowed_sig_sq: float,
                    configuration_report: str,
                    maploss,
                    scores,
                    config,
                    ) -> None:
    """
    Fits a parametric t-SNE model and optionally saves it to the desired directory.
    Fits either regular or multi-scale t-SNE
    :param model: nn.Module instance
    :param input_points: tensor of original points
    :param opt: optimizer instance
    :param perplexity: perplexity of a model. If passed None, multi-scale parametric t-SNE
    model will be trained
    :param n_epochs: Number of epochs for training
    :param dev: device for tensors (e.g. "cpu" or "cuda")
    :param save_dir_path: path to directory to save a trained model to
    :param epochs_to_save_after: number of epochs to save a model after. If passed None,
    model won't be saved at all
    :param early_exaggeration: Number of first training cycles in which
    exaggeration will be applied
    :param early_exaggeration_constant: Constant by which p_joint is multiplied in early exaggeration
    :param batch_size: Batch size for training
    :param dist_func_name: Name of distance function for distance matrix.
    Possible names: "euc", "jaccard", "cosine"
    :param bin_search_tol: A small number - tolerance threshold for binary search
    :param bin_search_max_iter: Number of max iterations for binary search
    :param min_allowed_sig_sq: Minimum allowed value for the spread of any distribution
    in conditional probability matrix
    :param max_allowed_sig_sq: Maximum allowed value for the spread of any distribution
    in conditional probability matrix
    :param configuration_report: Config of the model in string form for report purposes
    :return:
    """
    model.train()
    # batches_passed = 0
    model_name = get_random_string(6)
    epoch_losses = []


    # Function operates with DataLoader
    dataset=embedding_score_data(input_points, scores)
    train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    maploss1=MapLoss().to(device(config.dev))

    opt1 = torch.optim.Adam(list(model.parameters())+list(maploss1.parameters()), **config.optimization_conf)

    for epoch in range(n_epochs):
        train_loss = 0
        total_map_loss = 0
        batches_passed = 0
        epoch_start_time = datetime.datetime.now()

        # For every batch
        for list_with_batch in tqdm(train_dl):
            orig_points_batch, batch_scores = list_with_batch
        

            # Calculate conditional probability matrix in higher-dimensional space for the batch

            # Regular parametric t-SNE
            if perplexity is not None:
                target_entropy = log2(perplexity)
                p_cond_in_batch = calculate_optimized_p_cond(orig_points_batch,
                                                             target_entropy,
                                                             dist_func_name,
                                                             bin_search_tol,
                                                             bin_search_max_iter,
                                                             min_allowed_sig_sq,
                                                             max_allowed_sig_sq,
                                                             dev)
                if p_cond_in_batch is None:
                    continue
                p_joint_in_batch = make_joint(p_cond_in_batch)

            # Multiscale parametric t-SNE
            else:
                max_entropy = round(log2(batch_size / 2))
                n_different_entropies = 0
                mscl_p_joint_in_batch = zeros(batch_size, batch_size).to(device(dev))
                for h in range(1, max_entropy):
                    p_cond_for_h = calculate_optimized_p_cond(orig_points_batch,
                                                              h,
                                                              dist_func_name,
                                                              bin_search_tol,
                                                              bin_search_max_iter,
                                                              min_allowed_sig_sq,
                                                              max_allowed_sig_sq,
                                                              dev)
                    if p_cond_for_h is None:
                        continue
                    n_different_entropies += 1

                    p_joint_for_h = make_joint(p_cond_for_h)

                    # TODO This fails if the last batch doesn't match the shape of mscl_p_joint_in_batch
                    mscl_p_joint_in_batch += p_joint_for_h

                p_joint_in_batch = mscl_p_joint_in_batch / n_different_entropies

            # print(p_joint_in_batch.shape)

            # Apply early exaggeration to the conditional probability matrix
            if early_exaggeration:
                p_joint_in_batch *= early_exaggeration_constant
                early_exaggeration -= 1

            batches_passed += 1
            opt1.zero_grad()

            # Calculate joint probability matrix in lower-dimensional space for the batch
            embeddings = model(orig_points_batch)
            q_joint_in_batch = get_q_joint(embeddings, "euc", alpha=1)


            # Calculate loss
            loss = loss_function(p_joint_in_batch, q_joint_in_batch)
            train_loss += loss.item()

            mloss=maploss1(embeddings, batch_scores, p_joint_in_batch)
            total_map_loss += mloss.item()
            loss=loss+0.125*mloss

            # Make an optimization step

            loss.backward()
            opt1.step()

        epoch_end_time = datetime.datetime.now()
        time_elapsed = epoch_end_time - epoch_start_time

        # Report loss for epoch
        average_loss = train_loss / batches_passed
        average_map_loss = total_map_loss / batches_passed
        epoch_losses.append(average_loss)
        print(f'====> Epoch: {epoch + 1}. Time {time_elapsed}. Average loss: {average_loss:.4f}', flush=True)
        print(f'====> Epoch: {epoch + 1}. Time {time_elapsed}. Average Map loss: {average_map_loss:.4f}', flush=True)
        print("a1: "+str(maploss1.a1.item()))
        print("a2: "+str(maploss1.a2.item()))
        print("b: "+str(maploss1.b.item()))

        # Save model and loss history if needed
        # save_path = os.path.join(save_dir_path, f"{model_name}_epoch_{epoch + 1}")
        save_path = "akrmap"+f"{model_name}_epoch_{epoch + 1}"
        if epochs_to_save_after is not None and (epoch + 1) % epochs_to_save_after == 0:
            torch.save(model, save_path + ".pt")
            with open(save_path + ".json", "w") as here:
                json.dump(json.loads(configuration_report), here)
            print('Model saved as %s' % save_path, flush=True)

        if epochs_to_save_after is not None and epoch == n_epochs - 1:
            epoch_losses = array(epoch_losses)
            loss_save_path = save_path + "_loss.npy"
            np_save(loss_save_path, epoch_losses)
            print("Loss history saved in", loss_save_path, flush=True)

    print("final kernel params")
    print("a1: "+str(maploss1.a1.item()))
    print("a2: "+str(maploss1.a2.item()))
    print("b: "+str(maploss1.b.item()))

    return save_path, maploss1.a1.item(), maploss1.a2.item(), maploss1.b.item()


