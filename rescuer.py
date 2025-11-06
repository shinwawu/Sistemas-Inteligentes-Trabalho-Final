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
        n_clusters: int = 3,
        env_config_path: str = "env_config.txt",
    ):
        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.map = None  # explorer will pass the map
        self.victims: Dict[int, Tuple[Tuple[int, int], list]] = (
            {}
        )  # list of found victims
        self._maps_parts: List = []
        self._victs_parts: List[Dict] = []
        self._finished_from: int = 0
        self.is_master = is_master
        self.total_explorers = total_explorers
        self.env_victims_path = env_victims_path
        self.env_config_path = env_config_path
        self.data_csv = data_csv
        self.modelo_sobr = modelo_sobr
        self.modelo_tria = modelo_tria
        self.n_clusters = n_clusters
        self._pos2id: Optional[Dict[Tuple[int, int], int]] = (
            None  # (x_abs,y_abs) -> id CSV
        )
        self._base_abs: Tuple[int, int] = carregar_base(self.env_config_path)
        self._cluster_paths: List[str] = []
        self._last_mapping_report: str = ""

        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)
        # Registro global de socorristas (para round-robin de clusters)
        Rescuer.registry.append(self)

    def recebe_mapa(self, explorer_name, part_map, part_victims):
        """
            o explorador entrega as info do mapa e vitimas que ele tem para os rescuers
        """
        self._maps_parts.append(part_map)
        self._victs_parts.append(part_victims)
        self._finished_from += 1

        print(
            f" o socorrista recebeu mapa do  {explorer_name} "
            f"as informacoes foram ({len(part_map.map_data)} células, e {len(part_victims)} vítimas)."
        )

        if self.is_master and self._finished_from >= self.total_explorers:
            print(
                "socorrista mestre vai juntar as informacoes..."
            )
            self._unify_maps_and_cluster()

#auxílio do gemini para debuggar
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
        base_x, base_y = self._base_abs
        x_abs = base_x + int(x_local)
        y_abs = base_y + int(y_local)
        return x_abs, y_abs

    def _predict_sobr_e_tri(
        self, victim_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Carrega modelos e devolve (sobr_prob [0..1], tri_int)."""
        if not os.path.exists(self.modelo_sobr):
            raise FileNotFoundError("Faltando 'modelo_sobrevivencia.pkl'.")
        if not os.path.exists(self.modelo_tria):
            raise FileNotFoundError("Faltando 'modelo_triagem.pkl'.")

        reg = joblib.load(self.modelo_sobr)  # regressão p/ prob. de sobrevivência
        clf = joblib.load(self.modelo_tria)  # classificador de triagem
        df = pd.read_csv(self.data_csv)

        def montar_X(modelo):
            if not hasattr(modelo, "feature_names_in_"):
                raise ValueError(
                    "Modelo sem 'feature_names_in_'. Re-treine salvando esse atributo."
                )
            cols = list(modelo.feature_names_in_)
            faltam = [c for c in cols if c not in df.columns]
            if faltam:
                raise ValueError(f"Dataset não tem as colunas do modelo: {faltam}")
            return df.iloc[victim_ids][cols].copy().reset_index(drop=True)

        sobr = reg.predict(montar_X(reg)).clip(0, 1).astype(float)
        tri = clf.predict(montar_X(clf)).astype(int)
        return sobr, tri

    # ------------------------------ Pipeline principal ------------------------------

    def _unify_maps_and_cluster(self) -> None:
        """Unifica mapas e vítimas, prediz rótulos, clusteriza, salva arquivos e imagem."""
        # 1) Unificar mapas
        from map import Map

        mapa_unico = Map()
        merged_cells = {}
        for mp in self._maps_parts:
            merged_cells.update(mp.map_data)
        mapa_unico.map_data = merged_cells
        self.map = mapa_unico

        # 2) Unificar vítimas (apenas as lidas)
        victims_merged: Dict[int, Tuple[Tuple[int, int], list]] = {}
        for d in self._victs_parts:
            victims_merged.update(d)
        self.victims = victims_merged

        # 3) Mapear posições locais -> IDs do CSV (via posição absoluta)
        self._load_pos2id()
        pos_abs: List[Tuple[int, int]] = []
        victim_ids: List[int] = []
        total_lidas = len(self.victims)
        ok_abs = ok_swap = falhas = 0

        #calcula a posicao relativas da vitimas em relacao a base
        for _, (pos_local, _vs) in self.victims.items():
            xl, yl = pos_local
            xa, ya = self.converte_absoluto(xl, yl)

            # tentativa direta (x,y)
            if (xa, ya) in self._pos2id:
                pos_abs.append((xa, ya))
                victim_ids.append(self._pos2id[(xa, ya)])
                ok_abs += 1
                continue

            # fallback (y,x) para pipelines que invertam
            if (ya, xa) in self._pos2id:
                pos_abs.append((ya, xa))
                victim_ids.append(self._pos2id[(ya, xa)])
                ok_swap += 1
                continue

            falhas += 1

        self._last_mapping_report = (
            f"Mapeamento vítimas lidas -> ids: tot={total_lidas}, "
            f"ok_abs={ok_abs}, ok_swap={ok_swap}, falhas={falhas}"
        )
        print("RESCUER (mestre): " + self._last_mapping_report)

        if not pos_abs:
            print("RESCUER (mestre): nenhuma vítima mapeada (verifique BASE/eixos).")
            return

        # 4) Predição de SOBR e TRI (T1)
        sobr, tri = self._predict_sobr_e_tri(
            victim_ids
        )  # sobr ∈ [0,1]; tri ∈ {0,1,2,...}

        # 5) Montar features de clusterização: posição normalizada + sobr/tri
        xs = np.array([x for x, _ in pos_abs], dtype=float)
        ys = np.array([y for _, y in pos_abs], dtype=float)
        x_den = max(1.0, xs.max() - xs.min())
        y_den = max(1.0, ys.max() - ys.min())
        x_norm = (xs - xs.min()) / x_den
        y_norm = (ys - ys.min()) / y_den

        tri = np.asarray(tri, dtype=float)
        tri_norm = tri / max(1.0, float(tri.max()))  # normaliza TRI para 0..1
        sobr = np.asarray(sobr, dtype=float)  # já 0..1

        # Pesos (ajuste estratégico, mantendo comportamento)
        w_pos, w_sobr, w_tri = 1.0, 0.8, 0.6
        X = np.column_stack(
            [w_pos * x_norm, w_pos * y_norm, w_sobr * sobr, w_tri * tri_norm]
        )

        # 6) K-Means
        k = int(max(1, min(self.n_clusters, len(X))))
        os.environ.setdefault("OMP_NUM_THREADS", "1")  # evita avisos de MKL
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        rotulos = km.fit_predict(X)

        # 7) Salvar cluster*.txt (id_vict, x_abs, y_abs, sobr, tri)
        grupos: Dict[int, List[Tuple[int, int, int, float, int]]] = {}
        for (x_abs, y_abs), vid, s, t, r in zip(
            pos_abs, victim_ids, sobr, tri.astype(int), rotulos
        ):
            grupos.setdefault(int(r), []).append(
                (int(vid), int(x_abs), int(y_abs), float(s), int(t))
            )

        self._cluster_paths = []
        for i, r in enumerate(sorted(grupos.keys()), start=1):
            path = f"cluster{i}.txt"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                for row in grupos[r]:
                    w.writerow(row)
            self._cluster_paths.append(path)
        print("RESCUER (mestre): arquivos gerados -> " + ", ".join(self._cluster_paths))

        # 8) Atribuir clusters aos socorristas (round-robin por grupo)
        resc_names = sorted([ag.NAME for ag in Rescuer.registry]) or [
            "RESCUER_1",
            "RESCUER_2",
            "RESCUER_3",
        ]
        self.assignments = {name: [] for name in resc_names}
        for idx_lbl, lbl in enumerate(sorted(grupos.keys())):
            destino = resc_names[idx_lbl % len(resc_names)]
            self.assignments[destino] += [
                row[0] for row in grupos[lbl]
            ]  # apenas os ids
        print("RESCUER (mestre): atribuição inicial (id_vict por agente):")
        for kname, lst in self.assignments.items():
            print(f"  - {kname}: {sorted(lst)}")

        # 9) Visual 2D dos clusters
        try:
            import matplotlib.pyplot as plt

            cmap = plt.cm.get_cmap("tab10", max(1, len(set(rotulos))))
            plt.figure(figsize=(9, 7))
            for i, r in enumerate(sorted(set(rotulos))):
                idx = np.where(rotulos == r)[0]
                plt.scatter(
                    xs[idx],
                    ys[idx],
                    s=24,
                    c=[cmap(i)],
                    label=f"cluster {r}",
                    alpha=0.9,
                    edgecolors="none",
                )

            bx, by = self._base_abs
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
            plt.title(
                "Clusters (K-Means) — posição ABS (x,y); features: x, y, sobr, tri"
            )
            plt.xlabel("x (abs)")
            plt.ylabel("y (abs)")
            plt.grid(True, ls=":", alpha=0.6)
            plt.legend(loc="best", fontsize=9)
            plt.tight_layout()
            plt.savefig("clusters_visual.png", dpi=140)
            plt.close()
            print("RESCUER (mestre): visual salvo em clusters_visual.png")
        except Exception as e:
            print("RESCUER (mestre): falha ao gerar visual dos clusters:", e)

        # Fica ocioso (não se move)
        self.set_state(VS.IDLE)

    # ------------------------------ Loop do agente ------------------------------

    def deliberate(self) -> bool:
        """Rescuer não se move nesta tarefa."""
        return False
