# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch
import math
from src.utils import AllReduce


logger = getLogger()


def init_msn_loss(
    num_views=1,
    tau=0.1,
    me_max=True,
    return_preds=False
):
    """
    Make unsupervised MSN loss

    :num_views: number of anchor views
    :param tau: cosine similarity temperature
    :param me_max: whether to perform me-max regularization
    :param return_preds: whether to return anchor predictions
    """
    softmax = torch.nn.Softmax(dim=1)

    def sharpen(p, T):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(query, supports, support_labels, temp=tau):
        """ Soft Nearest Neighbours similarity classifier """
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)
        return softmax(query @ supports.T / temp) @ support_labels

    def loss(
        anchor_views,
        target_views,
        prototypes,
        proto_labels,
        T=0.25,
        use_entropy=False,
        use_sinkhorn=False,
        sharpen=sharpen,
        snn=snn,
        ot=None,
        ot_train=True,
        ot_sharpen=False
    ):
        # Step 1: compute anchor predictions
        probs = snn(anchor_views, prototypes, proto_labels)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            rho = 1.
            if ot is not None:
                # P2OT: eps of the solver does the sharpening; optional extra sharpen for ablation.
                tp = snn(target_views, prototypes, proto_labels)
                targets, rho = ot(sharpen(tp, T=T) if ot_sharpen else tp, train=ot_train)
            else:
                targets = sharpen(snn(target_views, prototypes, proto_labels), T=T)
                if use_sinkhorn:
                    targets = distributed_sinkhorn(targets)
            targets = torch.cat([targets for _ in range(num_views)], dim=0)

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if me_max:
            avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
            rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))

        sloss = 0.
        if use_entropy:
            sloss = torch.mean(torch.sum(torch.log(probs**(-probs)), dim=1))

        # -- logging
        with torch.no_grad():
            num_ps = float(len(set(targets.argmax(dim=1).tolist())))
            max_t = targets.max(dim=1).values.mean()
            min_t = targets.min(dim=1).values.mean()
            col = targets.sum(0) / targets.sum().clamp_min(1e-12) * targets.shape[1]
            log_dct = {'np': num_ps, 'max_t': max_t, 'min_t': min_t, 'rho': rho,
                       'size_min': float(col.min()), 'size_max': float(col.max())}

        if return_preds:
            return loss, rloss, sloss, log_dct, targets

        return loss, rloss, sloss, log_dct

    return loss


@torch.no_grad()
def distributed_sinkhorn(Q, num_itr=3, use_dist=True):
    _got_dist = use_dist and torch.distributed.is_available() \
        and torch.distributed.is_initialized() \
        and (torch.distributed.get_world_size() > 1)

    if _got_dist:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    Q = Q.T
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if _got_dist:
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_itr):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if _got_dist:
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.T


class PPOT:
    """Progressive partial OT pseudo-labels (P2OT, Zhang, Ren & He, ICLR 2024, arXiv 2401.09266).

    Solves  min_Q <Q, -log P> + eps*H(Q)
            s.t. Q 1 <= 1/N (via a zero-cost virtual cluster), total mass rho,
                 cluster marginal pulled to rho/K by lam * KL (soft, so sizes may differ).
    Scaling iterations as in the authors' SemiCurrSinkhornKnopp: a = alpha/(M b), b = beta/(M^T a),
    b[:K] **= lam/(lam+eps); the virtual column (mass 1 - rho) stays exact.
    Rows of the returned plan sum to each day's selected mass (<= 1) and are used as soft targets.
    A FIFO queue of past target probabilities lets the solver see more days than one batch; only
    the current batch's rows are returned. Warm-starts b across calls (training only).
    """

    def __init__(self, eps=0.1, lam=1.0, rho0=0.1, total_steps=1, queue=0, num_iters=1000, tol=1e-6):
        self.eps, self.lam, self.rho0 = float(eps), float(lam), float(rho0)
        self.total_steps, self.qsize = max(int(total_steps), 1), int(queue)
        self.num_iters, self.tol = int(num_iters), float(tol)
        self.step, self.bank, self.b = 0, None, None

    def rho(self, step=None):
        """Eq. 7: rho0 + (1 - rho0) exp(-5 (1 - t/T)^2), 1 at t = T."""
        t = min((self.step if step is None else step) / self.total_steps, 1.0)
        return self.rho0 + (1. - self.rho0) * math.exp(-5. * (1. - t) ** 2)

    @torch.no_grad()
    def solve(self, P, rho, b0=None):
        n, k = P.shape
        # Kernel exp(-C/eps) with C = [-log P, 0]. Each row is shifted by its max (a per-row
        # constant, absorbed by the scaling a) to avoid underflow; the virtual column keeps cost 0.
        L = torch.log(P.double().clamp_min(1e-300)) / self.eps
        s = L.max(dim=1, keepdim=True).values
        M = torch.cat([torch.exp(L - s), torch.exp(-s)], 1)
        alpha = torch.full((n, 1), 1. / n, dtype=M.dtype, device=P.device)
        beta = torch.full((k + 1, 1), rho / k, dtype=M.dtype, device=P.device)
        beta[-1] = 1. - rho
        fi = self.lam / (self.lam + self.eps)
        b = torch.ones(k + 1, 1, dtype=M.dtype, device=P.device) if b0 is None else b0.clone()
        for _ in range(self.num_iters):
            a = alpha / (M @ b).clamp_min(1e-300)
            b_new = beta / (M.T @ a).clamp_min(1e-300)
            b_new[:-1] = b_new[:-1] ** fi
            err = torch.norm(b_new - b)
            b = b_new
            if err <= self.tol:
                break
        plan = n * a * M * b.T
        return plan[:, :-1].float(), b

    @torch.no_grad()
    def __call__(self, P, train=True):
        """P: (B, K) target probabilities. Returns (B, K) soft targets and the rho used."""
        if not train:
            return self.solve(P, 1.0)[0], 1.0
        rho = self.rho()
        allP = P if self.bank is None else torch.cat([P, self.bank])
        b0 = self.b if (self.b is not None and self.b.device == P.device) else None
        Q, self.b = self.solve(allP, rho, b0)
        if self.qsize > 0:
            self.bank = allP[:self.qsize].detach().clone()
        self.step += 1
        return Q[:len(P)], rho
