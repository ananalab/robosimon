"""Banc d'essai Simon en simulation MuJoCo : scène, environnement et expert scripté.

Le temps est compté en images de contrôle (30 Hz). Toutes les durées de la
section 2.4 sont converties en nombres entiers d'images pour éviter les
erreurs d'arrondi.
"""
import numpy as np
import mujoco

FPS = 30
SOUS_PAS = 20                      # 1/600 s par pas physique
T_REPOS, D_ON, D_OFF, DELAI, D_GO, D_FB = 30, 18, 9, 60, 6, 6
S_H = 6                            # pas d'échantillonnage de l'historique
COURSE = 0.008                     # course des boutons (m)
HAUT_BOUTON = 0.032                # dessus d'un bouton au repos (z)
REPOS = np.array([0.15, -0.02, 0.10])
BORNES = np.array([[-0.11, -0.11, 0.0], [0.18, 0.11, 0.15]])
CENTRES = np.array([[-0.045, 0.045], [0.045, 0.045], [-0.045, -0.035], [0.045, -0.035]])
GO_POS = (0.0, -0.088)
COULEURS = np.array([[1, .08, .08], [.1, .9, .1], [.15, .35, 1], [1, .9, .05], [1, 1, 1]])
ETEINT = np.array([.3, .3, .3, .3, .18])      # facteur d'assombrissement à l'extinction
NOMS = ["rouge", "vert", "bleu", "jaune", "GO"]
H_IMG, W_IMG = 240, 320

REPOS_PH, PRESENTATION, DELAI_PH, DEPART, EXECUTION = range(5)


def t_max(L):
    return int(round((3 + 2.5 * L) * FPS))


def chronologie(sequence, t_repos=T_REPOS, d_on=D_ON, d_off=D_OFF, delai=DELAI, d_go=D_GO):
    """Intervalles d'allumage (début, fin, led) en images depuis le début de l'épisode.
    led vaut 0 à 3 pour les boutons, 4 pour la LED de départ.
    Renvoie aussi l'instant de début de la phase d'exécution."""
    intervalles, t = [], t_repos
    for c in sequence:
        intervalles.append((t, t + d_on, c))
        t += d_on + d_off
    t += delai
    intervalles.append((t, t + d_go, 4))
    return intervalles, t + d_go


def _mjcf():
    boutons = ""
    for i, (x, y) in enumerate(CENTRES):
        boutons += f"""
      <body name="bouton{i}" pos="{x} {y} 0.02" gravcomp="1">
        <joint name="piston{i}" type="slide" axis="0 0 1" range="{-COURSE} 0" stiffness="50"
               springref="0.004" damping="1" limited="true"/>
        <geom name="piston{i}" type="cylinder" pos="0 0 0.006" size="0.03 0.006" mass="0.01" rgba=".2 .2 .2 1"/>
        <geom name="led{i}" type="cylinder" pos="0 0 0.0122" size="0.026 0.0004" contype="0" conaffinity="0"/>
      </body>"""
    act = "".join(f'<position joint="doigt_{a}" kp="400" ctrlrange="{lo} {hi}"/>'
                  for a, lo, hi in zip("xyz", BORNES[0], BORNES[1]))
    return f"""
<mujoco model="simon">
  <option timestep="{1 / (FPS * SOUS_PAS)}" integrator="implicitfast"/>
  <visual><global offwidth="{W_IMG}" offheight="{H_IMG}"/><quality shadowsize="0"/>
    <headlight ambient=".25 .25 .25" diffuse=".35 .35 .35" specular="0 0 0"/></visual>
  <worldbody>
    <light name="lampe" pos="0 -0.3 0.8" dir="0 0.35 -0.8" diffuse=".6 .6 .6" castshadow="false"/>
    <geom name="table" type="plane" size=".6 .6 .01" rgba=".55 .45 .35 1"/>
    <camera name="cam" pos="0.03 -0.26 0.29" xyaxes="1 0 0 0 0.746 0.666" fovy="38"/>
    <!-- le doigt est déclaré en premier : qpos[0:3] est la position de son extrémité -->
    <body name="doigt" gravcomp="1">
      <joint name="doigt_x" type="slide" axis="1 0 0" damping="8"/>
      <joint name="doigt_y" type="slide" axis="0 1 0" damping="8"/>
      <joint name="doigt_z" type="slide" axis="0 0 1" damping="8"/>
      <geom name="doigt" type="capsule" fromto="0 0 0.01 0 0 0.09" size="0.01" mass="0.03" rgba=".85 .75 .65 1"/>
    </body>
    <body name="plateau">
      <geom type="box" pos="0 0 0.01" size="0.095 0.105 0.01" rgba=".12 .12 .12 1" contype="2" conaffinity="2"/>
      <geom name="led4" type="cylinder" pos="{GO_POS[0]} {GO_POS[1]} 0.0204" size="0.014 0.0004"
            contype="0" conaffinity="0"/>{boutons}
    </body>
  </worldbody>
  <actuator>{act}</actuator>
</mujoco>"""


MODELE_XML = _mjcf()


class SimonEnv:
    """Environnement Simon. Interface de type Gymnasium : reset(seed, options), step(action).
    L'image n'est rendue que si on la demande (rendu=True), pour économiser le calcul
    pendant l'évaluation. `info` ne sert jamais d'entrée aux politiques."""

    def __init__(self, rendu_actif=True):
        self.model = mujoco.MjModel.from_xml_string(MODELE_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, H_IMG, W_IMG) if rendu_actif else None
        m = self.model
        self.g_led = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"led{i}") for i in range(5)]
        self.g_piston = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"piston{i}") for i in range(4)]
        self.g_doigt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "doigt")
        self.g_table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.b_plateau = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plateau")
        self.qadr_piston = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"piston{i}")]
                            for i in range(4)]
        self.lumiere0 = m.light_diffuse.copy()
        self.table0 = m.geom_rgba[self.g_table].copy()

    # ---------------------------------------------------------------- épisode
    def reset(self, seed=0, options=None):
        options = options or {}
        self.sequence = [int(c) for c in options["sequence"]]
        self.L = len(self.sequence)
        rng = np.random.default_rng(seed)
        m = self.model
        # randomisation légère : lumière ±15 %, plateau ±1 cm, teinte de la table
        m.light_diffuse[:] = self.lumiere0 * rng.uniform(0.85, 1.15)
        self.decalage = rng.uniform(-0.01, 0.01, 2)
        m.body_pos[self.b_plateau, :2] = self.decalage
        m.geom_rgba[self.g_table, :3] = np.clip(self.table0[:3] + rng.uniform(-0.08, 0.08, 3), 0, 1)
        mujoco.mj_resetData(m, self.data)
        self.data.qpos[:3] = REPOS
        self.data.ctrl[:] = REPOS
        mujoco.mj_forward(m, self.data)

        self.intervalles, self.debut_exec = chronologie(self.sequence)
        self.t = 0
        self.progres = 0
        self.pressions = []              # (instant en images, bouton)
        self.enfonce = [False] * 4
        self.compteur = [0] * 4          # sous-pas consécutifs au-delà du seuil
        self.contact = [False] * 4
        self.contact_engage = [False] * 4
        self.contacts_rates = 0
        self.fb = None                   # (bouton, fin de la rétroaction)
        self.resultat, self.erreur = None, None
        self._maj_led()
        return self._obs(options.get("rendu", True)), self._info()

    def centres_boutons(self):
        return CENTRES + self.decalage

    def phase(self, t=None):
        t = self.t if t is None else t
        if t < T_REPOS:
            return REPOS_PH
        fin_pres = T_REPOS + self.L * (D_ON + D_OFF)
        if t < fin_pres:
            return PRESENTATION
        if t < fin_pres + DELAI:
            return DELAI_PH
        if t < self.debut_exec:
            return DEPART
        return EXECUTION

    def etat_led(self, t=None):
        t = self.t if t is None else t
        led = np.zeros(5, dtype=np.uint8)
        for a, b, c in self.intervalles:
            if a <= t < b:
                led[c] = 1
        if self.fb is not None and t < self.fb[1]:
            led[self.fb[0]] = 1
        return led

    def _maj_led(self):
        self.led = self.etat_led()
        for i in range(5):
            f = 1.0 if self.led[i] else ETEINT[i]
            self.model.geom_rgba[self.g_led[i]] = (*(COULEURS[i] * f), 1)

    def _obs(self, rendu):
        obs = {"etat": self.data.qpos[:3].astype(np.float32).copy()}
        if rendu:
            self.renderer.update_scene(self.data, camera="cam")
            obs["image"] = self.renderer.render().copy()
        return obs

    def _info(self):
        return {"t": self.t, "phase": self.phase(), "led": self.led.copy(), "progres": self.progres,
                "pressions": list(self.pressions), "resultat": self.resultat, "erreur": self.erreur,
                "sequence": list(self.sequence), "debut_exec": self.debut_exec}

    def _detecter(self, instant):
        d = self.data
        touches = set()
        for k in range(d.ncon):
            c = d.contact[k]
            if c.geom1 == self.g_doigt and c.geom2 in self.g_piston:
                touches.add(self.g_piston.index(c.geom2))
            elif c.geom2 == self.g_doigt and c.geom1 in self.g_piston:
                touches.add(self.g_piston.index(c.geom1))
        for i in range(4):
            if i in touches:
                self.contact[i] = True
            elif self.contact[i]:
                self.contact[i] = False
                if not self.contact_engage[i]:
                    self.contacts_rates += 1
                self.contact_engage[i] = False
            prof = -d.qpos[self.qadr_piston[i]]
            if not self.enfonce[i]:
                self.compteur[i] = self.compteur[i] + 1 if prof > 0.6 * COURSE else 0
                if self.compteur[i] >= 2:
                    self.enfonce[i] = True
                    self.contact_engage[i] = True
                    self._pression(i, instant)
            elif prof < 0.3 * COURSE:
                self.enfonce[i] = False
                self.compteur[i] = 0

    def _pression(self, b, instant):
        if self.resultat is not None:
            return
        self.pressions.append((instant, b))
        if instant < self.debut_exec:
            self.resultat, self.erreur = "echec", "anticipation"
            return
        attendu = self.sequence[self.progres]
        if b != attendu:
            precedent = self.sequence[self.progres - 1] if self.progres > 0 else None
            self.resultat = "echec"
            self.erreur = "repetition" if b == precedent else "memoire"
            return
        self.progres += 1
        self.fb = (b, int(np.ceil(instant)) + D_FB)
        if self.progres == self.L:
            self.resultat = "succes"

    def step(self, action, rendu=True):
        cible = np.clip(np.asarray(action, dtype=np.float64), BORNES[0], BORNES[1])
        self.data.ctrl[:] = cible
        for k in range(SOUS_PAS):
            mujoco.mj_step(self.model, self.data)
            self._detecter(self.t + (k + 1) / SOUS_PAS)
            if self.resultat is not None:
                break
        self.t += 1
        if self.resultat is None and self.t - self.debut_exec >= t_max(self.L):
            self.resultat = "echec"
            self.erreur = "motrice" if self.contacts_rates >= 3 else "blocage"
        self._maj_led()
        termine = self.resultat is not None
        return self._obs(rendu), float(self.resultat == "succes"), termine, False, self._info()


# ---------------------------------------------------------------------- expert
def jerk_min(p0, p1, n):
    s = (np.arange(1, n + 1) / n)[:, None]
    return p0 + (p1 - p0) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


def plan_expert(env, rng):
    """Trajectoire de cibles cartésiennes de l'expert, une par image, de t = 0 jusqu'au
    temps limite. Immobile en pose de repos jusqu'au départ + temps de réaction.
    Renvoie aussi les instants de décision : débuts des transits d'un bouton vers un
    bouton différent (utilisés par l'indice d'inertie)."""
    attente = env.debut_exec + int(round(rng.uniform(0.2, 0.5) * FPS))
    cibles = [np.tile(REPOS, (attente, 1))]
    p = REPOS.copy()
    centres = env.centres_boutons()
    decisions = []
    for j, b in enumerate(env.sequence):
        if j > 0 and b != env.sequence[j - 1]:
            decisions.append(sum(len(c) for c in cibles))
        xy = centres[b] + rng.normal(0, 0.003, 2)
        dessus = np.array([*xy, HAUT_BOUTON + 0.04]) + rng.normal(0, 0.003, 3)
        contact = np.array([*xy, HAUT_BOUTON])
        enfonce = np.array([*xy, HAUT_BOUTON - 0.012])
        for q, duree in ((dessus, rng.uniform(0.6, 1.0)), (contact, 0.25), (enfonce, 0.15),
                         (enfonce, 0.1), (dessus, 0.3)):
            n = int(round(duree * FPS))
            cibles.append(jerk_min(p, q, n))
            p = q
    total = env.debut_exec + t_max(env.L) + 1
    cibles.append(np.tile(p, (max(0, total - sum(len(c) for c in cibles)), 1)))
    return np.concatenate(cibles), decisions


def episode_expert(seed, sequence, rendu=False, env=None, rng_expert=None):
    """Joue un épisode avec l'expert. Renvoie (images, etats, actions, infos)."""
    env = env or SimonEnv(rendu_actif=rendu)
    obs, info = env.reset(seed, {"sequence": sequence, "rendu": rendu})
    plan, _ = plan_expert(env, rng_expert or np.random.default_rng(seed + 10_000_000))
    images, etats, actions, infos = [], [], [], []
    termine = False
    while not termine:
        a = plan[env.t]
        images.append(obs.get("image"))
        etats.append(obs["etat"])
        actions.append(a.astype(np.float32))
        infos.append(info)
        obs, _, termine, _, info = env.step(a, rendu=rendu)
    infos.append(info)  # info finale (résultat)
    return images, np.array(etats), np.array(actions), infos
