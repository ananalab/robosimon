"""Module de mémoire, têtes ACT et Diffusion Policy, variantes B0, B1, M1, M1p (section 5).

Les politiques travaillent sur les plongements gelés de E_h (ResNet-18), précalculés.
Les têtes sont réimplémentées d'après les fichiers de modélisation de LeRobot
(modeling_act.py, modeling_diffusion.py) avec les réglages de la section 5.3.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simon_sim import BORNES, FPS, S_H

H_PRED, H_EXEC, K_MAX = 32, 16, 128
VARIANTES = ("B0", "B1", "M1", "M1p")


def normaliser(x):
    lo, hi = (torch.as_tensor(b, device=x.device, dtype=x.dtype) for b in BORNES)
    return 2 * (x - lo) / (hi - lo) - 1


def denormaliser(x):
    lo, hi = (torch.as_tensor(b, device=x.device, dtype=x.dtype) for b in BORNES)
    return (x + 1) / 2 * (hi - lo) + lo


def indices_grille(t, k_max=K_MAX):
    """Grille absolue de l'historique, vectorisée. t : (B,) instants courants (images).
    Renvoie tau (B, k_max) : k_max - 1 cases de grille (multiples de S_H, les plus récents)
    puis l'image courante en dernière case ; vide (B, k_max) : True pour le remplissage.
    Une case de grille égale à t est masquée (l'image courante est déjà la dernière case).
    La même fonction sert à l'entraînement et à l'inférence."""
    n = t // S_H + 1                                               # nombre de cases de grille <= t
    j = torch.arange(k_max - 1, device=t.device) + (n - (k_max - 1)).clamp(min=0)[:, None]
    tau = torch.cat([j * S_H, t[:, None]], 1)
    vide = torch.cat([(j >= n[:, None]) | (j * S_H == t[:, None]),
                      torch.zeros_like(t[:, None], dtype=torch.bool)], 1)
    return tau.clamp(max=t[:, None]), vide


class EncodageAge(nn.Module):
    """Encodage sinusoïdal de l'âge (en secondes) de chaque image d'historique."""

    def __init__(self, dim, periode_min=0.4, periode_max=120.0):
        super().__init__()
        demi = dim // 2
        i = torch.arange(demi) / (demi - 1)
        f_max, f_min = 2 * math.pi / periode_min, 2 * math.pi / periode_max
        self.register_buffer("freqs", f_max * (f_min / f_max) ** i)

    def forward(self, age):                       # age : (B, K)
        angles = age.unsqueeze(-1) * self.freqs   # (B, K, dim/2)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ModuleMemoire(nn.Module):
    def __init__(self, dim_entree, dim=256, n_couches=2, n_tetes=4, n_jetons=4, dropout=0.1, avec_etat=False):
        super().__init__()
        self.projection = nn.Linear(dim_entree, dim)
        self.proj_etat = nn.Linear(3, dim) if avec_etat else None       # M1p
        self.age = EncodageAge(dim)
        couche = nn.TransformerEncoderLayer(dim, n_tetes, dim_feedforward=4 * dim,
                                            dropout=dropout, batch_first=True, norm_first=True)
        self.encodeur = nn.TransformerEncoder(couche, n_couches, enable_nested_tensor=False)
        self.requetes = nn.Parameter(0.02 * torch.randn(n_jetons, dim))
        self.attention = nn.MultiheadAttention(dim, n_tetes, dropout=dropout, batch_first=True)
        self.norme = nn.LayerNorm(dim)

    def forward(self, plongements, age, vide, etats=None):
        # plongements : (B, K, dim_entree), sorties gelées de E_h
        # age : (B, K) en secondes ; vide : (B, K), True pour les positions de remplissage
        x = self.projection(plongements) + self.age(age)
        if self.proj_etat is not None:
            x = x + self.proj_etat(etats)
        x = self.encodeur(x, src_key_padding_mask=vide)
        q = self.requetes.unsqueeze(0).expand(x.shape[0], -1, -1)
        jetons, _ = self.attention(q, x, x, key_padding_mask=vide)
        return self.norme(jetons)                 # (B, n_jetons, dim)


class Memoire(nn.Module):
    """Jetons mémoire d'une variante (dimension 256), ou None pour B0.
    B1 : quatre jetons issus de la séquence et de sa longueur (one-hot), plus les quatre
    jetons du module d'historique (voir rapport : sans historique, l'indice de progression
    n'est pas observable lorsqu'un bouton se répète, et B1 ne borne plus M1)."""

    def __init__(self, variante, dim_entree, dim=256):
        super().__init__()
        self.variante = variante
        self.module = None if variante == "B0" else ModuleMemoire(dim_entree, dim, avec_etat=variante == "M1p")
        self.oracle = nn.Linear(4 * 4 + 3, 4 * dim) if variante == "B1" else None
        self.n_jetons = {"B0": 0, "B1": 8, "M1": 4, "M1p": 4}[variante]

    def forward(self, lot):
        if self.module is None:
            return None
        jetons = self.module(lot["hist"], lot["age"], lot["vide"], lot.get("hist_etat"))
        if self.oracle is not None:
            jetons = torch.cat([jetons, self.oracle(lot["oracle"]).view(len(jetons), 4, -1)], 1)
        return jetons


# ------------------------------------------------------------------------- ACT
class ACT(nn.Module):
    """Tête ACT : transformeur encodeur-décodeur et auto-encodeur variationnel conditionnel."""

    def __init__(self, variante, dim_entree, d=512, n_tetes=8, n_enc=4, n_dec=1, d_ff=3200,
                 latent=32, n_vae=4, dropout=0.1, beta=10.0):
        super().__init__()
        self.n_obs, self.beta, self.latent = 1, beta, latent
        self.memoire = Memoire(variante, dim_entree)
        enc = lambda n: nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, n_tetes, d_ff, dropout, activation="relu", batch_first=True), n, enable_nested_tensor=False)
        # auto-encodeur variationnel : [CLS, état, actions] -> (mu, log sigma^2)
        self.vae_cls = nn.Parameter(0.02 * torch.randn(1, 1, d))
        self.vae_etat = nn.Linear(3, d)
        self.vae_action = nn.Linear(3, d)
        self.vae_pos = nn.Parameter(0.02 * torch.randn(1, H_PRED + 2, d))
        self.vae = enc(n_vae)
        self.vae_sortie = nn.Linear(d, 2 * latent)
        # encodeur principal : [latent, état, jetons mémoire, jeton visuel]
        self.proj_latent = nn.Linear(latent, d)
        self.proj_etat = nn.Linear(3, d)
        self.proj_image = nn.Linear(dim_entree, d)
        self.proj_memoire = nn.Linear(256, d)
        self.pos_enc = nn.Parameter(0.02 * torch.randn(1, 3 + self.memoire.n_jetons, d))
        self.encodeur = enc(n_enc)
        self.decodeur = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, n_tetes, d_ff, dropout, activation="relu", batch_first=True), n_dec)
        self.pos_dec = nn.Parameter(0.02 * torch.randn(1, H_PRED, d))
        self.sortie = nn.Linear(d, 3)

    def _encoder(self, lot, z):
        jetons = [self.proj_latent(z)[:, None], self.proj_etat(lot["etat"][:, -1])[:, None]]
        mem = self.memoire(lot)
        if mem is not None:
            jetons.append(self.proj_memoire(mem))
        jetons.append(self.proj_image(lot["cur"][:, -1])[:, None])
        return self.encodeur(torch.cat(jetons, 1) + self.pos_enc)

    def _decoder(self, memoire):
        q = self.pos_dec.expand(len(memoire), -1, -1)
        return self.sortie(self.decodeur(torch.zeros_like(q) + q, memoire))

    def perte(self, lot):
        B = len(lot["action"])
        x = torch.cat([self.vae_cls.expand(B, -1, -1), self.vae_etat(lot["etat"][:, -1])[:, None],
                       self.vae_action(lot["action"])], 1) + self.vae_pos
        mu, logvar = self.vae_sortie(self.vae(x)[:, 0]).chunk(2, -1)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        pred = self._decoder(self._encoder(lot, z))
        l1 = F.l1_loss(pred, lot["action"])
        kld = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(-1).mean()
        return l1 + self.beta * kld, {"l1": l1.item(), "kld": kld.item()}

    @torch.no_grad()
    def agir(self, lot, generateur=None):
        z = torch.zeros(len(lot["etat"]), self.latent, device=lot["etat"].device)
        return self._decoder(self._encoder(lot, z))

    def optimiseur(self, lr=1e-5):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4), None


# ---------------------------------------------------------- Diffusion Policy
class EmbPasDiffusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, k):
        demi = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(demi, device=k.device) / (demi - 1))
        a = k.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], -1)


class BlocConv1d(nn.Sequential):
    def __init__(self, c_in, c_out, k, n_groupes=8):
        super().__init__(nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.GroupNorm(n_groupes, c_out), nn.Mish())


class BlocResiduelFiLM(nn.Module):
    """Bloc résiduel conditionnel de Chi et al. : modulation FiLM en échelle et en biais."""

    def __init__(self, c_in, c_out, dim_cond, k=5):
        super().__init__()
        self.b1, self.b2 = BlocConv1d(c_in, c_out, k), BlocConv1d(c_out, c_out, k)
        self.cond = nn.Sequential(nn.Mish(), nn.Linear(dim_cond, 2 * c_out))
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.c_out = c_out

    def forward(self, x, cond):
        h = self.b1(x)
        echelle, biais = self.cond(cond).unsqueeze(-1).chunk(2, 1)
        h = self.b2(echelle * h + biais)
        return h + self.res(x)


class UNet1D(nn.Module):
    """U-Net temporel conditionnel (ConditionalUnet1D de Chi et al., 2023)."""

    def __init__(self, dim_cond_global, dims=(256, 512, 1024), k=5, dim_pas=128):
        super().__init__()
        self.pas = nn.Sequential(EmbPasDiffusion(dim_pas), nn.Linear(dim_pas, 4 * dim_pas), nn.Mish(),
                                 nn.Linear(4 * dim_pas, dim_pas))
        dc = dim_pas + dim_cond_global
        chaine = [3, *dims]
        self.desc = nn.ModuleList()
        for i, (a, b) in enumerate(zip(chaine[:-1], chaine[1:])):
            dernier = i == len(dims) - 1
            self.desc.append(nn.ModuleList([BlocResiduelFiLM(a, b, dc, k), BlocResiduelFiLM(b, b, dc, k),
                                            nn.Conv1d(b, b, 3, 2, 1) if not dernier else nn.Identity()]))
        self.milieu = nn.ModuleList([BlocResiduelFiLM(dims[-1], dims[-1], dc, k) for _ in range(2)])
        self.mont = nn.ModuleList()
        for a, b in zip(reversed(dims[1:]), reversed(dims[:-1])):     # comme Chi et al. : deux remontées
            self.mont.append(nn.ModuleList([BlocResiduelFiLM(2 * a, b, dc, k), BlocResiduelFiLM(b, b, dc, k),
                                            nn.ConvTranspose1d(b, b, 4, 2, 1)]))
        self.final = nn.Sequential(BlocConv1d(dims[0], dims[0], k), nn.Conv1d(dims[0], 3, 1))

    def forward(self, x, k, cond):                 # x : (B, H, 3)
        cond = torch.cat([self.pas(k), cond], -1)
        x = x.transpose(1, 2)
        pile = []
        for r1, r2, bas in self.desc:
            x = r2(r1(x, cond), cond)
            pile.append(x)
            x = bas(x)
        for r in self.milieu:
            x = r(x, cond)
        for r1, r2, haut in self.mont:
            x = r2(r1(torch.cat([x, pile.pop()], 1), cond), cond)
            x = haut(x)
        return self.final(x).transpose(1, 2)


def betas_cosinus(n, s=0.008):
    """Calendrier squaredcos_cap_v2 (Nichol et Dhariwal, 2021), comme dans diffusers."""
    f = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return torch.tensor([min(1 - f((i + 1) / n) / f(i / n), 0.999) for i in range(n)], dtype=torch.float32)


class DP(nn.Module):
    def __init__(self, variante, dim_entree, n_obs=2, dim_image=256, n_pas=100, n_pas_inference=10,
                 dims=(256, 512, 1024), n_passe=0):
        super().__init__()
        self.n_obs, self.n_pas, self.n_inf = n_obs, n_pas, n_pas_inference
        # PTP (Torne et al., 2025) : la trajectoire débruitée inclut aussi les n_passe actions
        # passées ; elles ne servent qu'à l'entraînement et sont ignorées à l'exécution.
        self.n_passe = n_passe
        self.memoire = Memoire(variante, dim_entree)
        self.proj_image = nn.Linear(dim_entree, dim_image)
        dim_cond = n_obs * (dim_image + 3) + 256 * self.memoire.n_jetons
        self.unet = UNet1D(dim_cond, dims)
        alphas_cum = torch.cumprod(1 - betas_cosinus(n_pas), 0)
        self.register_buffer("alphas_cum", alphas_cum)

    def _cond(self, lot):
        c = [torch.cat([self.proj_image(lot["cur"]), lot["etat"]], -1).flatten(1)]
        mem = self.memoire(lot)
        if mem is not None:
            c.append(mem.flatten(1))
        return torch.cat(c, -1)

    def perte(self, lot):
        a0 = lot["action"]
        bruit = torch.randn_like(a0)
        k = torch.randint(0, self.n_pas, (len(a0),), device=a0.device)
        ac = self.alphas_cum[k][:, None, None]
        pred = self.unet(ac.sqrt() * a0 + (1 - ac).sqrt() * bruit, k, self._cond(lot))
        l = F.mse_loss(pred, bruit)
        return l, {"mse": l.item()}

    @torch.no_grad()
    def agir(self, lot, generateur=None):
        """Échantillonnage DDIM déterministe (eta = 0) en n_inf pas, écrêtage à ±1."""
        cond = self._cond(lot)
        B = len(cond)
        x = torch.randn(B, self.n_passe + H_PRED, 3, device=cond.device, generator=generateur)
        pas = list(range(0, self.n_pas, self.n_pas // self.n_inf))[::-1]
        for i, k in enumerate(pas):
            ac = self.alphas_cum[k]
            ac_prec = self.alphas_cum[pas[i + 1]] if i + 1 < len(pas) else torch.tensor(1.0, device=x.device)
            eps = self.unet(x, torch.full((B,), k, device=x.device), cond)
            x0 = ((x - (1 - ac).sqrt() * eps) / ac.sqrt()).clamp(-1, 1)
            eps = (x - ac.sqrt() * x0) / (1 - ac).sqrt()
            x = ac_prec.sqrt() * x0 + (1 - ac_prec).sqrt() * eps
        return x[:, self.n_passe:]

    def optimiseur(self, lr=1e-4, n_iter=20_000, prechauffage=500):
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6)
        f = lambda i: (i + 1) / prechauffage if i < prechauffage else \
            0.5 * (1 + math.cos(math.pi * (i - prechauffage) / max(1, n_iter - prechauffage)))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, f)


def creer(famille, variante, dim_entree):
    """ACT, DP, DPptp (DP avec prédiction des 16 actions passées) ou DPo1 (DP avec une seule
    image d'observation, comme ACT : pas de vitesse lisible dans l'observation)."""
    if famille == "ACT":
        return ACT(variante, dim_entree)
    return DP(variante, dim_entree, n_passe=16 if famille == "DPptp" else 0, n_obs=1 if famille == "DPo1" else 2)


# --------------------------------------------------------- lots d'entraînement
def oracle_onehot(seq, L):
    """seq : (B, 4) avec -1 en remplissage ; L : (B,). Renvoie (B, 19)."""
    oh = F.one_hot(seq.clamp(min=0).long(), 4).float() * (seq >= 0).float()[..., None]
    return torch.cat([oh.flatten(1), F.one_hot((L - 2).long(), 3).float()], -1)


class Donnees:
    """Jeu de démonstrations résident sur le GPU ; construit les lots à la volée."""

    def __init__(self, d, device, k_max=None):
        self.dev = device
        self.plong = torch.as_tensor(d["plongements"], device=device)              # fp16
        self.etat = normaliser(torch.as_tensor(d["etat"], device=device, dtype=torch.float32))
        self.action = normaliser(torch.as_tensor(d["action"], device=device, dtype=torch.float32))
        self.debut = torch.as_tensor(d["debut_episode"], device=device, dtype=torch.long)
        self.seq = torch.as_tensor(d["sequence"], device=device, dtype=torch.long)
        self.L = (self.seq >= 0).sum(1)
        n = len(self.plong)
        self.episode = torch.repeat_interleave(torch.arange(len(self.debut) - 1, device=device),
                                               self.debut[1:] - self.debut[:-1])
        self.t = torch.arange(n, device=device) - self.debut[self.episode]
        longueur_max = int((self.debut[1:] - self.debut[:-1]).max())
        self.k_max = k_max or min(K_MAX, longueur_max // S_H + 2)

    def lot(self, idx, n_obs=1, avec_action=True, passe=0):
        """idx : indices globaux d'images. Construit toutes les entrées possibles ;
        chaque politique ne lit que celles de sa variante."""
        e, t = self.episode[idx], self.t[idx]
        a, fin = self.debut[e], self.debut[e + 1]
        obs = torch.stack([a + (t - k).clamp(min=0) for k in range(n_obs - 1, -1, -1)], 1)   # (B, n_obs)
        tau, vide = indices_grille(t, self.k_max)
        g = a[:, None] + tau
        lot = {"cur": self.plong[obs].float(), "etat": self.etat[obs],
               "hist": self.plong[g].float(), "hist_etat": self.etat[g],
               "age": (t[:, None] - tau).float() / FPS, "vide": vide,
               "oracle": oracle_onehot(self.seq[e], self.L[e])}
        if avec_action:
            j = idx[:, None] + torch.arange(-passe, H_PRED, device=self.dev)
            j = torch.maximum(torch.minimum(j, fin[:, None] - 1), a[:, None])
            lot["action"] = self.action[j]
        return lot


if __name__ == "__main__":
    # vérifications rapides de la grille
    tau, vide = indices_grille(torch.tensor([0, 5, 6, 13, 1000]), 128)
    assert tau[0, -1] == 0 and vide[0, :-1].all()
    assert (tau[1][~vide[1]] == torch.tensor([0, 5])).all()
    assert (tau[2][~vide[2]] == torch.tensor([0, 6])).all()
    assert (tau[3][~vide[3]] == torch.tensor([0, 6, 12, 13])).all()
    assert (~vide[4]).sum() == 128 and tau[4, -2] == 996 and tau[4, 0] == 996 - 126 * 6
    for fam in ("ACT", "DP"):
        for v in VARIANTES:
            m = creer(fam, v, 512)
            print(fam, v, sum(p.numel() for p in m.parameters()) / 1e6, "M paramètres")
    print("ok")
