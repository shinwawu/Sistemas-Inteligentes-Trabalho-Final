##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position
import time
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

import random

from queue import PriorityQueue

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


##  ---    cria uma classe para empilhar as posições já visitadas para poder voltar

"""   ???      Ou poderia tbm voltar pelo próprio mapa que passou para os socorristas? 
tinha pensado nisso, mas tava tentando essa lógica aqui primeir                      ???     """

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0


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


        ##   ----  criei essas variáveis a mais, mas ainda não terminei  ----
  
        self.custo_volta = 1.05  # margem para custo de volta
        self.margem_seguranca = 1.575  # margem de seguranca

        self.walk_stack = Stack() 

        self.x = 0
        self.y = 0
        self.current_index = 1

        self.flag = True

    def recebe_mapa(self, explorer_name, part_map, part_victims):
        """
        o explorador entrega as info do mapa e vitimas que ele tem para os rescuers
        """
        self.pedaco_mapa.append(part_map)
        # with open(r'C:\Users\User\Desktop\SI_finalproject\Sistemas-Inteligentes-Trabalho-Final\data.csv',
        #            "w") as f:
        

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

        xs = np.array([x for x, _ in pos_abs], dtype=float)
        ys = np.array([y for _, y in pos_abs], dtype=float)
        x_den = max(1.0, xs.max() - xs.min())
        y_den = max(1.0, ys.max() - ys.min())
        x_norm = (xs - xs.min()) / x_den
        y_norm = (ys - ys.min()) / y_den

        tri = np.asarray(tri, dtype=float)
        tri_norm = tri / max(1.0, float(tri.max()))
        sobr = np.asarray(sobr, dtype=float)

        w_pos, w_sobr, w_tri = 1.0, 0.8, 0.6
        X = np.column_stack(
            [w_pos * x_norm, w_pos * y_norm, w_sobr * sobr, w_tri * tri_norm]
        )

        k = 3
        os.environ.setdefault(
            "OMP_NUM_THREADS", "1"
        )  # recomendacao do gemini para o erro de num threads
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        rotulos = kmeans.fit_predict(X)

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

        resc_names = sorted([ag.NAME for ag in Rescuer.registry]) or [
            "RESCUER_1",
            "RESCUER_2",
            "RESCUER_3",
        ]
        self.assignments = {name: [] for name in resc_names}
        for idx_lbl, lbl in enumerate(sorted(grupos.keys())):
            destino = resc_names[idx_lbl % len(resc_names)]
            self.assignments[destino] += [row[0] for row in grupos[lbl]]
        print("socorrista mestre: atribuiu:")
        for kname, lst in self.assignments.items():
            print(f"  - {kname}: {sorted(lst)}")

        try:
            # sugestao da visualizacao foi feita com auxilio do gemini
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

        self.set_state(VS.ACTIVE)


        # self.go_save_victims(mapa_unificado, vitimas_unificadas)
    
    def calcula_custo(self, bx, by):
        """
        Calcula o custo mínimo real para retornar para a base,
        a partir da posição atual.
        Base = (0, 0)
        """
        x, y = self.x, self.y

        #volta para base, senão calcula outro destino que não seja a base
        if bx == None or by == None:
            bx, by = 0, 0

        dx = abs(bx - x)
        dy = abs(by - y)

        # número de passos diagonais possíveis
        diag = min(dx, dy)

        # passos restantes em linha reta
        linha = abs(dx - dy)

        # custo total mínimo
        return diag * self.COST_DIAG + linha * self.COST_LINE
    
    def get_next_position(self):

        ##  ---  escolhe o mapa  de acordo com o socorrista específico  ----
        if self.NAME == 'RESCUER_1':
            position = self.pedaco_mapa[0]
        elif self.NAME == 'RESCUER_2':
            position = self.pedaco_mapa[1]
        else:
            position = self.pedaco_mapa[2]

        # position = self.pedaco_mapa[2]

        # print("\n--- DIR ---")
        # print(dir(position))

        # print("\n--- DICT ---")
        # print(position.__dict__)

        # time.sleep(10)
        # chaves são (x,y)
        keys = list(position.map_data.keys())

        # acabou o mapa
        if self.current_index >= len(keys):
            print(' >>>>>>>>>>>>>>>>>    A   C  A   B  O  U      O      M  A  P   A  >>>>>>>>>>>>>>>>>')
            return None

        # pega a próxima posição (x, y)
        prox = keys[self.current_index]

        # avança o índice
        self.current_index += 1

        return prox


    def come_back(self, to_pos=None):
        """
        agente volta.
        """
        if to_pos is None:
            # se tiver vazio, so volta
            if self.walk_stack.is_empty():
                return

            # tira da stack a ultima move
            dx, dy = self.walk_stack.pop()

            # inverte p voltar
            voltar_x, voltar_y = -dx, -dy

            # Executa o movimento de retorno
            resultado = self.walk(voltar_x, voltar_y)

            # realiza a volta
            if resultado == VS.EXECUTED:
                self.x += voltar_x
                self.y += voltar_y
            return

    
    def go_save_victms(self):
        prox = self.get_next_position()
        if prox is None:
            print('RETORNOU -------- //////////////////////////')
            return None

        px, py = prox

        # converte posição absoluta para movimento
        dx = px - self.x
        dy = py - self.y
        
        # caso o movimento seja parado
        if (dx, dy) == (0, 0):
            if not self.walk_stack.is_empty():
                # print('\nBOOOOMMP :: ', self.x, self.y)
                # print(last_dx, last_dy, '\n')
                last_dx, last_dy = self.walk_stack.pop()
             
                result = self.walk(-last_dx, -last_dy)
                if result == VS.EXECUTED:
                    self.x -= last_dx
                    self.y -= last_dy
                    
            # print('\nBOOOOMMP   SEM REMOVER NADA DA FILA :: ', self.x, self.y)
            return True

        if abs(dx) > 2 or abs(dy) > 2:
            print('\n\n HOUVE UM ERRO AQUI MEU\n\n')
            print(f'{self.NAME}_ pos_atual :: ', self.x, self.y, '   prox ::', prox)
        
            print('dx:: ', dx, 'dy::', dy)
            
        else:
            print(f'{self.NAME}_ pos_atual :: ', self.x, self.y, '   prox ::', prox)

        # executa o passo
        resultado = self.walk(dx, dy)

        # guarda histórico
        if resultado == VS.EXECUTED:
            self.x += dx
            self.y += dy
            self.walk_stack.push((self.x, self.y))
            return True
        else:
            print(f"{self.NAME} walk failed in ({self.x}, {self.y})")
        # print('pilha:: ', self.walk_stack.items)


    def continuar_explorando(self):
        """
        Verifica se há bateria suficiente para:
        - continuar explorando
        - + voltar para a base com margem de segurança
        """

        # Custo mínimo de volta AGORA
        custo_volta = self.calcula_custo(None, None)
        print('custo voltar :: ', self.calcula_custo(None, None), 'tempo restante ::: ', self.get_rtime())
        # quanto tempo resta
        temp_rest = self.get_rtime()

        # margem de segurança (ex.: 1.3 = 30% extra)
        custo_necessario = custo_volta * self.margem_seguranca

        return temp_rest >= custo_necessario

    def deliberate(self) -> bool:
        # print('\n\n <<<<<  I N I C I A N D O   S O C O R R I S T A S >>>>> \n\n')
        # Se ainda há fronteira e pode explorar -> explorar
        # self.aestrela()
        # return False
        if self.flag and self.continuar_explorando():
            # print(f'{self.NAME}antes:: ', self.get_rtime())
            resp = self.go_save_victms()
            if resp is None:
                self.flag = False

            # print(f'{self.NAME}depois:: ', self.get_rtime())
            return True

        print(f'SOCORRISTA {self.NAME} voltando para a base')

        ##   ----     Essa lógica copiei tbm, mas ainda não terminei, pq não consegui 
        # entender direito como que usa o 'frontier'    ----
        if (self.x, self.y) != (0, 0):
        
            self.come_back()
            
            return True

        print(f'{self.NAME} FINISHED')
        return False


    def h_score(self, destino):
        """Calcula o h_score faltante"""
        # self.calcula_custo()
        linha = destino[0]
        coluna = destino[1]

        return abs(self.x - coluna) + abs(self.y - linha)

    def aestrela(self):
        print(self.map)
        print(self.map.__dict__)
        # f_score = {celula: float("inf") for celula in self.map[0].map_data.keys()}
        # print(f_score)
        # time.sleep()
        # caminho = ""
        # return caminho

        pass

