##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position

import os
import csv
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from vs.abstract_agent import AbstAgent
from vs.constants import VS
import joblib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =========================== Utilitários de IO ===========================


def carregar_base(path: str) -> Tuple[int, int]:
    """le a posição da base no x,y do arquivo env_config.txt"""
    if not os.path.exists(path):
        return 0, 0
    basex, basey = 0, 0
    with open(path, "r") as arq:
        for linha in arq:
            string = linha.strip()
            if not string or string.startswith("#"):
                continue
            if string.upper().startswith("BASE"):
                partes = string.split()
                if len(partes) >= 2 and "," in partes[1]:
                    x_str, y_str = partes[1].split(",")
                    try:
                        bx, by = int(x_str), int(y_str)
                    except Exception:
                        pass
                break
    return bx, by


## Classe que define o Agente Rescuer com um plano fixo
# A classe foi ajustada e debbugada com ajuda da LLM
class Rescuer(AbstAgent):
    """Recebe mapas/vítimas dos exploradores, unifica, prediz (sobr/tri),
    faz K-Means, grava cluster*.txt e gera imagem clusters_visual.png."""

    registry: List["Rescuer"] = []

    def __init__(
        self,
        env,
        config_file,
        is_master: bool = False,
        total_explorers: int = 3,
        env_victims_path: str = "env_victims.txt",
        data_csv: str = "data.csv",
        modelo_sobr: str = "modelo_sobrevivencia.pkl",
        modelo_tria: str = "modelo_triagem.pkl",
        env_config_path: str = "env_config.txt",
    ):
        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.map = None  # explorer will pass the map
        self.victims: Dict[int, Tuple[Tuple[int, int], list]] = (
            {}
        )  # list of found victims
        self.pedaco_mapa: List = []
        self._victs_parts: List[Dict] = []
        self.finalizado: int = 0
        self.is_master = is_master
        self.total_explorers = total_explorers
        self.env_victims_path = env_victims_path
        self.env_config_path = env_config_path
        self.data_csv = data_csv
        self.modelo_sobr = modelo_sobr
        self.modelo_tria = modelo_tria
        self._pos2id: Optional[Dict[Tuple[int, int], int]] = (
            None  # (x_abs,y_abs) -> id CSV
        )
        self.base_pos_abs: Tuple[int, int] = carregar_base(self.env_config_path)
        self.cluster_paths: List[str] = []
        self.resumo_mapeamento: str = ""

        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)
        # Registro global de socorristas (para round-robin de clusters)
        Rescuer.registry.append(self)

    def recebe_mapa(self, explorer_name, part_map, part_victims):
        """
        o explorador entrega as info do mapa e vitimas que ele tem para os rescuers
        """
        self.pedaco_mapa.append(part_map)
        self._victs_parts.append(part_victims)
        self.finalizado += 1

        print(
            f" o socorrista recebeu mapa do  {explorer_name} "
            f"as informacoes foram ({len(part_map.map_data)} células, e {len(part_victims)} vítimas)."
        )

        if self.is_master and self.finalizado >= self.total_explorers:
            print("socorrista mestre vai juntar as informacoes...")
            self.mapeamento_clusterizacao()

    # auxílio do gemini para debuggar
    def _load_pos2id(self) -> None:
        """Carrega mapeamento (x_abs, y_abs) -> id (linha do data.csv) a partir de env_victims.txt.
        O arquivo usa 'linha,coluna'; aqui padronizamos (x=col, y=lin)."""
        if self._pos2id is not None:
            return
        pos2id: Dict[Tuple[int, int], int] = {}
        with open(self.env_victims_path, "r", newline="") as f:
            for idx, line in enumerate(f):
                s = line.strip()
                if not s:
                    continue
                lin, col = map(int, s.split(","))  # arquivo: linha,coluna
                x, y = col, lin  # nosso padrão: x=coluna, y=linha
                pos2id[(x, y)] = idx
        self._pos2id = pos2id

    def converte_absoluto(self, x_local: int, y_local: int) -> Tuple[int, int]:
        """
        calcula as coordenadas locais do explorador para coordenadas absolutas do mapa, somando a posição da base.
        """
        base_x, base_y = self.base_pos_abs
        x_abs = base_x + int(x_local)
        y_abs = base_y + int(y_local)
        return x_abs, y_abs

    def predict(self, victim_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """carrega os meus modelos salvos que estao em pkl e faz predict"""
        if not os.path.exists(self.modelo_sobr):
            raise FileNotFoundError("falta modelo_sobrevivencia.pkl")
        if not os.path.exists(self.modelo_tria):
            raise FileNotFoundError("falta modelo_triagem.pkl")

        reg = joblib.load(self.modelo_sobr)
        clf = joblib.load(self.modelo_tria)
        df = pd.read_csv(self.data_csv)

        def montar_X(modelo):
            cols = list(modelo.feature_names_in_)
            return df.iloc[victim_ids][cols].copy().reset_index(drop=True)

        sobr = reg.predict(montar_X(reg)).clip(0, 1).astype(float)
        tri = clf.predict(montar_X(clf)).astype(int)
        return sobr, tri

    # Funcao feita com auxilio do Gemini Pro, estava com dificuldade de mapear e debuggar, foi
    def mapeamento_clusterizacao(self) -> None:
        """junta as informacoes, clusteriza, salva arquivos e imagem."""
        from map import Map

        mapa_unico = Map()
        celulas_junt = {}
        for mp in self.pedaco_mapa:
            celulas_junt.update(mp.map_data)
        mapa_unico.map_data = celulas_junt
        self.map = mapa_unico

        vitimas_junt: Dict[int, Tuple[Tuple[int, int], list]] = {}
        for d in self._victs_parts:
            vitimas_junt.update(d)
        self.victims = vitimas_junt

        self._load_pos2id()
        pos_abs: List[Tuple[int, int]] = []
        victim_ids: List[int] = []
        total_lidas = len(self.victims)
        ok_abs = ok_swap = falhas = 0

        for _, (pos_local, _vs) in self.victims.items():
            xl, yl = pos_local
            xa, ya = self.converte_absoluto(xl, yl)

            if (xa, ya) in self._pos2id:
                pos_abs.append((xa, ya))
                victim_ids.append(self._pos2id[(xa, ya)])
                ok_abs += 1
                continue

            if (ya, xa) in self._pos2id:
                pos_abs.append((ya, xa))
                victim_ids.append(self._pos2id[(ya, xa)])
                ok_swap += 1
                continue

            falhas += 1

        self.resumo_mapeamento = (
            f"vitimas lidas -> ids: tot={total_lidas}, "
            f"ok_abs={ok_abs}, ok_swap={ok_swap}, falhas={falhas}"
        )
        print(self.resumo_mapeamento)

        if not pos_abs:
            print("nenhuma vítima mapeada")
            return
        # até aqui foi feito com auxílio da LLM
        sobr, tri = self.predict(victim_ids)

        xs = np.array([p[0] for p in pos_abs], dtype=float).ravel()
        ys = np.array([p[1] for p in pos_abs], dtype=float).ravel()

        x_den = max(1.0, xs.max() - xs.min())
        y_den = max(1.0, ys.max() - ys.min())
        x_norm = (xs - xs.min()) / x_den
        y_norm = (ys - ys.min()) / y_den

        tri = np.asarray(tri, dtype=float)
        tri_norm = tri / max(1.0, float(tri.max()))
        sobr = np.asarray(sobr, dtype=float)

        w_pos, w_sobr, w_tri = 1.0, 0.8, 0.6
        X = np.column_stack([x_norm, y_norm])

        k = 3
        os.environ.setdefault(
            "OMP_NUM_THREADS", "1"
        )  # recomendacao do gemini para o erro de num threads
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        rotulos = kmeans.fit_predict(X)
        rotulos = np.asarray(rotulos, dtype=int).ravel()
        grupos: Dict[int, List[Tuple[int, int, int, float, int]]] = {}
        for (x_abs, y_abs), vid, s, t, r in zip(
            pos_abs, victim_ids, sobr, tri.astype(int), rotulos
        ):
            grupos.setdefault(int(r), []).append(
                (int(vid), int(x_abs), int(y_abs), float(s), int(t))
            )

        self.cluster_paths = []
        for i, r in enumerate(sorted(grupos.keys()), start=1):
            path = f"cluster{i}.txt"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                for row in grupos[r]:
                    w.writerow(row)
            self.cluster_paths.append(path)
        print("arquivos gerados".join(self.cluster_paths))

        # ============================
        #  ATRIBUIÇÃO POR DISTÂNCIA
        # ============================

        socorristas = Rescuer.registry
        nomes = [ag.NAME for ag in socorristas]
        self.assignments = {name: [] for name in nomes}

        # posição de cada socorrista
        pos_soc = {ag.NAME: ag.base_pos_abs for ag in socorristas}

        def distancia(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        # calcular centroide de cada cluster
        cluster_inf = {}
        for lbl, lista in grupos.items():
            xs = [x for (_, x, y, s, t) in lista]
            ys = [y for (_, x, y, s, t) in lista]
            centro = (np.mean(xs), np.mean(ys))
            cluster_inf[lbl] = centro

        # ======== ATRIBUIÇÃO 1-TO-1 POR DISTÂNCIA ========
        for lbl, lista in grupos.items():

            centro = cluster_inf[lbl]

            melhor = None
            melhor_dist = float("inf")

            for nome in nomes:
                d = distancia(centro, pos_soc[nome])
                if d < melhor_dist:
                    melhor_dist = d
                    melhor = nome

            # atribui TODAS as vítimas do cluster ao socorrista escolhido
            self.assignments[melhor] += [row[0] for row in lista]

        # print do resultado
        print("\nAtribuição por distância:")
        for nome, vitimas in self.assignments.items():
            print(f"  - {nome}: {len(vitimas)} vítimas")

        # ----- visualização -----
        try:
            # garante que xs, ys e rótulos são arrays 1D NumPy
            xs_plot = np.asarray(xs, dtype=float).ravel()
            ys_plot = np.asarray(ys, dtype=float).ravel()
            rotulos_plot = np.asarray(rotulos, dtype=int).ravel()

            # força todos a terem o mesmo tamanho
            n = min(xs_plot.size, ys_plot.size, rotulos_plot.size)
            xs_plot = xs_plot[:n]
            ys_plot = ys_plot[:n]
            rotulos_plot = rotulos_plot[:n]

            cmap = plt.cm.get_cmap("tab10", 10)
            plt.figure(figsize=(9, 7))

            for r in sorted(set(rotulos_plot)):
                r_int = int(r)
                idx = np.where(rotulos_plot == r_int)[0]

                # filtra qualquer índice fora do tamanho válido (defensivo)
                idx = idx[idx < xs_plot.size]

                # evitar clusters vazios
                if idx.size == 0:
                    continue

                plt.scatter(
                    xs_plot[idx],
                    ys_plot[idx],
                    s=24,
                    c=cmap(r_int % 10),
                    label=f"cluster {r_int}",
                    alpha=0.9,
                    edgecolors="none",
                )

            bx, by = self.base_pos_abs
            plt.scatter(
                [bx],
                [by],
                s=120,
                marker="*",
                facecolors="none",
                edgecolors="k",
                linewidths=1.5,
                label=f"BASE ({bx},{by})",
            )

            plt.title("kmeans")
            plt.xlabel("x (abs)")
            plt.ylabel("y (abs)")
            plt.grid(True, ls=":", alpha=0.6)
            plt.legend(loc="best", fontsize=9)
            plt.tight_layout()
            plt.savefig("clusters_visual.png", dpi=140)
            plt.close()
            print("clusterizacao salva visual")

        except Exception as e:
            print("falha no salvamento visual", e)

    # ------------------------------------------------------------------

    def deliberate(self) -> bool:
        return False
