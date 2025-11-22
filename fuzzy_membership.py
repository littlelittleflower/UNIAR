import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _trapezoid_params(raw: torch.Tensor):
    deltas = F.softplus(raw) + 1e-4
    a = deltas[..., 0]
    b = a + deltas[..., 1]
    c = b + deltas[..., 2]
    d = c + deltas[..., 3]
    return a, b, c, d

def _triangular_params(raw: torch.Tensor):
    deltas = F.softplus(raw) + 1e-4
    a = deltas[..., 0]
    b = a + deltas[..., 1]
    c = b + deltas[..., 2]
    return a, b, c

def _positive(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x) + 1e-6

class TrapezoidalFuzzy(nn.Module):
    def __init__(self, init=(10.,20.,30.,40.)):
        super().__init__()
        init_raw = torch.tensor([init[0], init[1]-init[0], init[2]-init[1], init[3]-init[2]], dtype=torch.float32)
        self.raw = nn.Parameter(init_raw)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a,b,c,d = _trapezoid_params(self.raw)
        mu = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        rise = (x > a) & (x < b)
        mu[rise] = (x[rise] - a) / (b - a + 1e-8)
        top = (x >= b) & (x <= c)
        mu[top] = 1.0
        fall = (x > c) & (x < d)
        mu[fall] = (d - x[fall]) / (d - c + 1e-8)
        return mu.clamp(0., 1.)
    def params(self):
        return _trapezoid_params(self.raw)
    def compare_ge(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu_x = self.forward(x)
        mu_y = self.forward(y)
        return (mu_x * (1 - (1 - mu_y))).clamp(0., 1.)

class FuzzyRelationMatrix(nn.Module):
    def __init__(self, num_rel: int, init_sigma: float=0.25):
        super().__init__()
        self.num_rel = num_rel
        rank = max(4, num_rel // 8)
        self.U = nn.Parameter(torch.randn(num_rel, rank) * 0.1)
        self.V = nn.Parameter(torch.randn(num_rel, rank) * 0.1)
        self.m_raw = nn.Parameter(torch.zeros(1))
        self.s_raw = nn.Parameter(torch.tensor(math.log(math.exp(init_sigma)-1.0)))
    def forward(self, C: torch.Tensor) -> torch.Tensor:
        assert C.dim()==2 and C.size(0)==C.size(1)==self.num_rel
        normC = C.float()
        normC = normC / (normC.max() + 1e-8)
        center = torch.sigmoid(self.U @ self.V.t())
        m = (center + torch.sigmoid(self.m_raw)).clamp(0., 1.)
        sigma = _positive(self.s_raw) + 1e-6
        M = torch.exp( - (normC - m).pow(2) / (2 * sigma * sigma) )
        return M.clamp(0., 1.)

class TriangularFuzzy(nn.Module):
    def __init__(self, init=(1.,3.,5.)):
        super().__init__()
        init_raw = torch.tensor([init[0], init[1]-init[0], init[2]-init[1]], dtype=torch.float32)
        self.raw = nn.Parameter(init_raw)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a,b,c = _triangular_params(self.raw)
        mu = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        rise = (x > a) & (x < b)
        mu[rise] = (x[rise] - a) / (b - a + 1e-8)
        mu[(x >= b) & (x <= c)] = 1.0
        fall = (x > c) & (x < (c + (c-b)))
        mu[fall] = ((c + (c-b)) - x[fall]) / ((c - b) + 1e-8)
        return mu.clamp(0., 1.)
    def params(self):
        return _triangular_params(self.raw)
    def forward_with_shift(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        a,b,c = _triangular_params(self.raw)
        a = a + delta; b = b + delta; c = c + delta
        mu = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        rise = (x > a) & (x < b)
        mu[rise] = (x[rise] - a) / (b - a + 1e-8)
        mu[(x >= b) & (x <= c)] = 1.0
        fall = (x > c) & (x < (c + (c-b)))
        mu[fall] = ((c + (c-b)) - x[fall]) / ((c - b) + 1e-8)
        return mu.clamp(0., 1.)

class GaussianFuzzy(nn.Module):
    def __init__(self, init_mean: float=0.5, init_sigma: float=0.25):
        super().__init__()
        self.m = nn.Parameter(torch.tensor(init_mean, dtype=torch.float32))
        self.s_raw = nn.Parameter(torch.tensor(math.log(math.exp(init_sigma)-1.0), dtype=torch.float32))
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(self.s_raw) + 1e-6
        return torch.exp( - (z - self.m).pow(2) / (2 * sigma * sigma) ).clamp(0., 1.)

def tnorm_mix(mu_rel: torch.Tensor, mu_time: torch.Tensor, mu_attr: torch.Tensor, kind: str="product") -> torch.Tensor:
    if kind == "min":
        return torch.minimum(torch.minimum(mu_rel, mu_time), mu_attr)
    elif kind == "lukasiewicz":
        return torch.clamp(mu_rel + mu_time + mu_attr - 2.0, min=0.0, max=1.0)
    else:
        return (mu_rel * mu_time * mu_attr).clamp(0., 1.)

# Simple DGL helpers as strings (to avoid importing dgl in this environment)
def attach_uniform_mu(g, weight_key: str="weight"):
    w = g.edata.get(weight_key, None)
    if w is None:
        w = torch.ones(g.num_edges(), device=g.device)
        g.edata[weight_key] = w
    g.edata['mu_rel']  = w.clone()
    g.edata['mu_time'] = w.clone()
    g.edata['mu_attr'] = w.clone()
    if 'eid' not in g.edata:
        g.edata['eid'] = torch.arange(g.num_edges(), device=g.device)
    return g

