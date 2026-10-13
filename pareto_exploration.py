"""
Pareto exploration (Frank-Wolfe MGDA / PCGrad / FAMO) for
  - specialist ResNet18 : 2 tasks  = per-class loss ("not d", "d")
  - generalist ResNet50 : 10 tasks = per-digit BCE loss

Paste into your notebook/script. It assumes these already exist in your code:
  DEVICE, BATCH_SIZE, SPEC_EPOCHS, GEN_EPOCHS, make_resnet18, BinaryMNIST,
  spec_train_base, train_loader (generalist), specialists_predict
"""
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =====================================================================
# 0. SHARED BUILDING BLOCKS
# =====================================================================

def flatten_grads(grads):
    """Concatenate a list of per-parameter gradients into one flat vector."""
    return torch.cat([g.reshape(-1) for g in grads])


def task_gradients(losses, params, normalize=True):
    """
    Build the gradient matrix G of shape [K tasks, P params].
    One autograd.grad call per task; retain_graph keeps the graph alive
    so the next task (and later a normal backward) can reuse it.
    """
    rows = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        # parameters that do not affect this task get a zero gradient
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        rows.append(flatten_grads(grads))
    G = torch.stack(rows)                                   # [K, P]
    norms = G.norm(dim=1, keepdim=True).clamp_min(1e-12)    # [K, 1]
    if normalize:
        # unit-norm gradients so a big-magnitude task cannot dominate alpha
        G = G / norms
    return G.detach(), norms.squeeze(1).detach()


@torch.no_grad()
def frank_wolfe_simplex(M, pref=None, lam=0.0, iters=20, tol=1e-6):
    """
    Frank-Wolfe solver on the probability simplex.
        min_alpha  alpha^T M alpha  +  lam * ||alpha - pref||^2
    - lam = 0         -> plain MGDA (one single min-norm point)
    - lam > 0 + pref  -> preference-guided MGDA (sweeping pref traces the front)
    Only the KxK Gram matrix M = G G^T is needed, so this is very cheap.
    """
    K = M.shape[0]
    alpha = torch.full((K,), 1.0 / K, device=M.device)      # start from uniform weights
    if pref is None:
        pref, lam = alpha.clone(), 0.0                      # no preference -> no proximal term
    pref = pref.to(M.device).float()

    # Hessian of the objective (constant, because the objective is quadratic)
    H = 2 * M + 2 * lam * torch.eye(K, device=M.device)

    for _ in range(iters):
        grad = H @ alpha - 2 * lam * pref                   # gradient of the objective
        j = torch.argmin(grad)                              # linear minimization oracle: best vertex e_j
        d = -alpha.clone()
        d[j] += 1.0                                         # direction d = e_j - alpha
        gap = -(grad @ d)                                   # Frank-Wolfe duality gap
        if gap < tol:                                       # converged
            break
        # exact line search for a quadratic, clipped to [0, 1] to stay inside the simplex
        gamma = (gap / (d @ H @ d).clamp_min(1e-12)).clamp(0.0, 1.0)
        alpha = alpha + gamma * d                           # convex combination keeps alpha on simplex
    return alpha


def combine_mgda_fw(G, norms, pref=None, lam=0.0):
    """MGDA with Frank-Wolfe: d = sum_i alpha_i g_i, alpha from the solver."""
    alpha = frank_wolfe_simplex(G @ G.T, pref=pref, lam=lam)
    # rescale: we solved on unit-norm gradients, so restore a sensible step size
    d = (alpha[:, None] * G).sum(0) * norms.mean()
    return d, alpha


def combine_pcgrad(G, norms):
    """
    PCGrad: for each task gradient, remove the component that conflicts
    (negative dot product) with every other task gradient, then average.
    """
    K = G.shape[0]
    G_proj = G.clone()
    for i in range(K):
        order = [j for j in range(K) if j != i]
        random.shuffle(order)                               # random order as in the paper
        for j in order:
            dot = G_proj[i] @ G[j]
            if dot < 0:                                     # conflict -> project onto normal plane of g_j
                G_proj[i] = G_proj[i] - dot / (G[j] @ G[j]).clamp_min(1e-12) * G[j]
    d = G_proj.mean(0) * norms.mean()
    return d, torch.full((K,), 1.0 / K, device=G.device)


class FAMO:
    """
    FAMO: O(1) per-step multi-task weighting. Keeps learnable logits xi;
    weights = softmax(xi) are pushed toward tasks whose loss is decreasing slowly.
    No per-task backward pass needed (only one backward on a scalar loss).
    Note: the loss-decrease signal here uses consecutive minibatch losses
    (cheap, but noisier than the paper's extra forward pass).
    """
    def __init__(self, K, device, lr=0.025, gamma=1e-3):
        self.xi = torch.zeros(K, device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.xi], lr=lr, weight_decay=gamma)
        self.prev = None

    def weights(self):
        return F.softmax(self.xi, dim=0).detach()

    def weighted_loss(self, losses):
        """Scalar loss whose gradient equals c * sum_i z_i * grad(L_i) / L_i."""
        L = losses.clamp_min(1e-8)
        z = self.weights()
        c = (1.0 / (z / L.detach()).sum()).detach()
        return c * (torch.log(L) * z / L.detach()).sum()

    def update(self, losses):
        """Update xi from the loss decrease since the previous step."""
        cur = losses.detach().clamp_min(1e-8)
        if self.prev is not None:
            delta = torch.log(self.prev) - torch.log(cur)   # >0 means that task is improving
            with torch.enable_grad():
                z = F.softmax(self.xi, dim=0)
                self.xi.grad = torch.autograd.grad(z, self.xi, grad_outputs=delta)[0]
            self.opt.step()
        self.prev = cur


def assign_flat_grad(params, d):
    """Write the combined flat direction d back into each param.grad."""
    off = 0
    for p in params:
        n = p.numel()
        p.grad = d[off:off + n].view_as(p).clone()
        off += n


def pareto_front(points, maximize=True):
    """
    Return indices of non-dominated points.
    points: list of equal-length lists/tensors of objective values.
    A point is dominated if another is >= everywhere and > somewhere (if maximize).
    """
    P = torch.tensor([list(map(float, p)) for p in points])
    if not maximize:
        P = -P
    keep = []
    for i in range(len(P)):
        dominated = any(
            bool((P[j] >= P[i]).all() and (P[j] > P[i]).any())
            for j in range(len(P)) if j != i
        )
        if not dominated:
            keep.append(i)
    return keep


def _selftest():
    """Sanity check: Frank-Wolfe must match the closed-form 2-task MGDA solution."""
    g1, g2 = torch.tensor([1.0, 0.2]), torch.tensor([0.1, 1.0])
    G = torch.stack([g1, g2])
    alpha = frank_wolfe_simplex(G @ G.T, iters=200, tol=1e-10)
    closed = ((g2 @ (g2 - g1)) / ((g1 - g2) @ (g1 - g2))).clamp(0, 1)   # alpha for g1
    assert abs(alpha[0].item() - closed.item()) < 1e-3, (alpha, closed)
    assert abs(alpha.sum().item() - 1) < 1e-5 and (alpha >= 0).all()
    print("FW self-test passed:", alpha.tolist())


# =====================================================================
# 1. SPECIALIST RESNET18  (K = 2 tasks: "not d" loss, "d" loss)
# =====================================================================

def train_specialist(digit, method="mgda_fw", pref=None, lam=0.5, tag=""):
    """
    method : none | mgda_fw | pcgrad | famo
       none = your original fixed class weights [1, 9] (baseline)
    pref   : (r_neg, r_pos) preference on the simplex, used by mgda_fw
    lam    : strength of the preference (0 = pure MGDA)
    """
    model = make_resnet18(num_outputs=2).to(DEVICE)
    train_loader_s = DataLoader(BinaryMNIST(spec_train_base, digit),
                                batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    params = [p for p in model.parameters() if p.requires_grad]

    pref_t = torch.tensor(pref, dtype=torch.float32) if pref is not None else None
    famo = FAMO(2, DEVICE) if method == "famo" else None
    fixed_w = torch.tensor([1.0, 9.0], device=DEVICE)       # baseline weights

    for epoch in range(SPEC_EPOCHS):
        model.train()
        total_loss, alpha_sum, alpha_n = 0.0, torch.zeros(2), 0

        for images, labels in train_loader_s:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)                          # forward pass
            ce = F.cross_entropy(outputs, labels, reduction="none")   # per-sample, unweighted

            # ---- split the loss into the two tasks (per-class) ----
            pos, neg = labels == 1, labels == 0
            L_neg = ce[neg].mean() if neg.any() else None
            L_pos = ce[pos].mean() if pos.any() else None

            optimizer.zero_grad()

            if method == "none":
                # baseline: fixed class weights, same as nn.CrossEntropyLoss(weight=[1, 9])
                w = fixed_w[labels]
                loss = (ce * w).sum() / w.sum()
                loss.backward()
                alpha = torch.tensor([1.0, 9.0]) / 10.0

            elif L_neg is None or L_pos is None:
                # guard: batch missing a class -> only one task exists, nothing to balance
                loss = L_neg if L_pos is None else L_pos
                loss.backward()
                alpha = torch.tensor([1.0, 0.0]) if L_pos is None else torch.tensor([0.0, 1.0])

            elif method == "famo":
                losses = torch.stack([L_neg, L_pos])
                famo.update(losses)                          # adapt task weights from loss trend
                loss = famo.weighted_loss(losses)            # one scalar backward, O(1) cost
                loss.backward()
                alpha = famo.weights().cpu()

            else:
                # ---- gradient-based methods: need one gradient per task ----
                losses = [L_neg, L_pos]
                G, norms = task_gradients(losses, params)    # G: [2, P]
                if method == "mgda_fw":
                    d, alpha = combine_mgda_fw(G, norms, pref=pref_t, lam=lam)
                elif method == "pcgrad":
                    d, alpha = combine_pcgrad(G, norms)
                else:
                    raise ValueError(method)
                assign_flat_grad(params, d)                  # combined direction -> param.grad
                loss = (alpha[0] * L_neg + alpha[1] * L_pos).detach()

            optimizer.step()                                 # Adam unchanged, only grads differ
            total_loss += float(loss)
            alpha_sum += alpha.detach().cpu().float()
            alpha_n += 1

        a = (alpha_sum / alpha_n).tolist()
        print(f"  specialist {digit}{tag} | epoch {epoch+1}/{SPEC_EPOCHS} | "
              f"loss {total_loss/len(train_loader_s):.4f} | alpha(neg,pos)=({a[0]:.3f},{a[1]:.3f})")

    model.eval()
    return model


@torch.no_grad()
def eval_specialist(model, val_loader):
    """Return (false-positive rate, false-negative rate) on binary validation data."""
    model.eval()
    fp = fn = neg = pos = 0
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        pred = model(images).argmax(1)
        fp += ((pred == 1) & (labels == 0)).sum().item()
        fn += ((pred == 0) & (labels == 1)).sum().item()
        neg += (labels == 0).sum().item()
        pos += (labels == 1).sum().item()
    return fp / max(neg, 1), fn / max(pos, 1)


def explore_specialist_pareto(digit, val_loader, n_points=7, method="mgda_fw", lam=0.5):
    """
    Train one specialist per preference r_pos in [0.1, 0.9], evaluate (FPR, FNR),
    keep the non-dominated ones, and save checkpoints + metadata.
    """
    results = []
    for k, r_pos in enumerate(torch.linspace(0.1, 0.9, n_points).tolist()):
        pref = (1 - r_pos, r_pos)                            # preference over (neg, pos)
        model = train_specialist(digit, method=method, pref=pref, lam=lam, tag=f" p{k}")
        fpr, fnr = eval_specialist(model, val_loader)
        torch.save(model.state_dict(), f"specialist_{digit}_p{k}.pt")
        results.append({"k": k, "pref": pref, "fpr": fpr, "fnr": fnr})
        print(f"  -> point {k}: pref={pref}  FPR={fpr:.4f}  FNR={fnr:.4f}")

    # both objectives are errors, so minimize -> maximize=False
    keep = pareto_front([[r["fpr"], r["fnr"]] for r in results], maximize=False)
    for i, r in enumerate(results):
        r["pareto"] = i in keep
    with open(f"specialist_{digit}_pareto.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pareto set for digit {digit}: points {keep}")
    return results


# =====================================================================
# 2. GENERALIST RESNET50  (K = 10 tasks: one BCE per digit)
# =====================================================================
# Cost control: gradients are taken ONLY w.r.t. the pooled features (input of
# the fc head) = MGDA "shared representation" upper-bound trick. Then a single
# normal backward goes through the whole network.
# Requires a torchvision-style model with .avgpool and .fc (10 outputs).

def train_generalist(model, saveFile, method="mgda_fw", pref=None, lam=0.5,
                     solve_every=1, make_optimizer=None):
    """
    method      : none | mgda_fw | pcgrad | famo
    pref        : length-10 preference vector on the simplex (mgda_fw)
    solve_every : re-solve alpha every N steps, reuse in between (saves time)
    """
    model = model.to(DEVICE)
    optimizer = (make_optimizer or (lambda p: torch.optim.Adam(p, lr=1e-4)))(model.parameters())
    K = 10
    pref_t = torch.tensor(pref, dtype=torch.float32) if pref is not None else None
    famo = FAMO(K, DEVICE) if method == "famo" else None

    # forward hook captures the pooled features inside the graph (no extra forward)
    store = {}
    hook = model.avgpool.register_forward_hook(lambda m, i, o: store.__setitem__("f", o))

    alpha, step = torch.full((K,), 1.0 / K, device=DEVICE), 0
    for epoch in range(GEN_EPOCHS):
        model.train()
        run, alpha_sum, n = 0.0, torch.zeros(K), 0

        for images, targets, _, img_a, img_b in train_loader:
            images = images.to(DEVICE)

            # ---- teacher labels from the two specialists (unchanged logic) ----
            with torch.no_grad():
                soft = torch.maximum(specialists_predict(img_a.to(DEVICE)),
                                     specialists_predict(img_b.to(DEVICE)))
                soft = (soft >= 0.5).float()                 # binarize teacher output

            logits = model(images)                           # [B, 10]
            feats = store["f"]                               # [B, 2048, 1, 1], part of the graph

            # ---- per-digit tasks: mean BCE over the batch for each column ----
            losses = F.binary_cross_entropy_with_logits(logits, soft, reduction="none").mean(0)  # [10]

            optimizer.zero_grad()

            if method == "none":
                loss = losses.mean()                         # baseline = your original BCE

            elif method == "famo":
                famo.update(losses)                          # O(1) weight update from loss trend
                loss = famo.weighted_loss(losses)
                alpha = famo.weights()

            elif method == "mgda_fw":
                if step % solve_every == 0:
                    # task gradients only at the feature layer (cheap, passes only through fc)
                    G, _ = task_gradients(list(losses), [feats])
                    # Frank-Wolfe on the 10x10 Gram matrix -> alpha on the simplex
                    pt = pref_t.to(DEVICE) if pref_t is not None else None
                    alpha = frank_wolfe_simplex(G @ G.T, pref=pt, lam=lam)
                loss = (alpha.detach() * losses).sum()       # single weighted backward below

            elif method == "pcgrad":
                G, norms = task_gradients(list(losses), [feats])
                d, alpha = combine_pcgrad(G, norms)          # de-conflicted feature gradient
                d_feat = d.view_as(feats)
                # backbone is driven by the surrogate (feats . d_feat);
                # the fc head trains on the plain mean BCE using detached features
                head_loss = F.binary_cross_entropy_with_logits(
                    model.fc(feats.detach().flatten(1)), soft)
                loss = (feats * d_feat.detach()).sum() + head_loss
            else:
                raise ValueError(method)

            loss.backward()                                  # one backward through ResNet50
            optimizer.step()

            step += 1
            run += losses.mean().item()
            alpha_sum += alpha.detach().cpu().float()
            n += 1

        a = (alpha_sum / n).numpy().round(3).tolist()
        print(f"Epoch {epoch+1}/{GEN_EPOCHS} | teacher loss {run/n:.4f} | alpha {a}")

    hook.remove()
    torch.save(model.state_dict(), saveFile)
    return model


@torch.no_grad()
def eval_generalist(model, val_loader, threshold=0.5):
    """Per-digit F1 vs the true multi-label targets -> tensor [10]."""
    model.eval()
    tp = torch.zeros(10); fp = torch.zeros(10); fn = torch.zeros(10)
    for batch in val_loader:
        images, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
        pred = (torch.sigmoid(model(images)) >= threshold).float()
        tp += (pred * targets).sum(0).cpu()
        fp += (pred * (1 - targets)).sum(0).cpu()
        fn += ((1 - pred) * targets).sum(0).cpu()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def make_preferences(K=10, n_points=6, seed=0):
    """Uniform preference first, then random Dirichlet points to cover the simplex."""
    g = torch.Generator().manual_seed(seed)
    prefs = [torch.full((K,), 1.0 / K)]
    for _ in range(n_points - 1):
        prefs.append(torch.distributions.Dirichlet(torch.ones(K)).sample())
    return [p.tolist() for p in prefs]


def explore_generalist_pareto(make_model, val_loader, n_points=6, method="mgda_fw", lam=0.5):
    """
    One generalist per preference vector -> per-digit F1 vector (10 objectives)
    -> keep non-dominated checkpoints. make_model() must return a fresh ResNet50.
    """
    results = []
    for k, pref in enumerate(make_preferences(10, n_points)):
        model = make_model()
        train_generalist(model, f"generalist_p{k}.pt", method=method, pref=pref, lam=lam)
        f1 = eval_generalist(model, val_loader)
        results.append({"k": k, "pref": pref, "f1_per_digit": f1.tolist(),
                        "f1_mean": f1.mean().item()})
        print(f"  -> point {k}: mean F1 {f1.mean():.4f}")

    keep = pareto_front([r["f1_per_digit"] for r in results], maximize=True)
    for i, r in enumerate(results):
        r["pareto"] = i in keep
    with open("generalist_pareto.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pareto set (generalist): points {keep}")
    return results


# =====================================================================
# USAGE
# =====================================================================
# _selftest()                                              # verify solver first
# explore_specialist_pareto(digit=3, val_loader=spec_val_loader)
# explore_generalist_pareto(lambda: make_resnet50_10out(), gen_val_loader)
# train_generalist(model, "gen.pt", method="mgda_fw")      # single run, no sweep
