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

        self.x, self.y = self.base_pos_abs
        
        self.plan = []  
        self.my_victims = []
        self.victim_locs = {} 
        self.current_goal = None


        ##   ----  criei essas variáveis a mais, mas ainda não terminei  ----
  
        self.custo_volta = 1.05  # margem para custo de volta
        self.margem_seguranca = 1.575  # margem de seguranca

        self.walk_stack = Stack() 
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
        if self._pos2id is not None:
            return
            
        pos2id: Dict[Tuple[int, int], int] = {}
        
        # Lê o arquivo diretamente como X, Y
        with open(self.env_victims_path, "r", newline="") as f:
            for idx, line in enumerate(f):
                s = line.strip()
                if not s:
                    continue
                # PDF: "x a coluna e y a linha"
                x_str, y_str = s.split(",")
                x, y = int(x_str), int(y_str)
                pos2id[(x, y)] = idx
                
        self._pos2id = pos2id

    def converte_absoluto(self, x_local: int, y_local: int) -> Tuple[int, int]:
        """
        calcula as coordenadas locais do explorador para coordenadas absolutas do mapa, somando a posição da base.
        """
        bx, by = self.base_pos_abs
        x_abs = bx + int(x_local)
        y_abs = by + int(y_local)
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

    def calcular_custo_sequencia(self, sequencia):
        """Calcula a distância total de uma sequência de vítimas"""
        custo_total = 0
        atual_pos = (self.x, self.y) # Começa onde o agente está (Base)
        
        for vid in sequencia:
            dest_pos = self.victim_locs[vid]
            # Usa heurística (distância direta) para ser rápido. 
            # Rodar A* aqui deixaria o AG muito lento.
            dist = self.heuristic(atual_pos, dest_pos)
            custo_total += dist
            atual_pos = dest_pos
            
        # Adiciona custo de volta à base
        custo_total += self.heuristic(atual_pos, self.base_pos_abs)
        return custo_total

    def executar_ag(self):
        """AG para ordenar my_victims maximizando o fitness."""
        if len(self.my_victims) < 2:
            return

        pop_size = 20
        generations = 30
        
        # População Inicial
        pop = [self.my_victims[:]] # Inclui a original
        for _ in range(pop_size - 1):
            ind = self.my_victims[:]
            random.shuffle(ind)
            pop.append(ind)
            
        for g in range(generations):
            # Ordena pelo Fitness (Decrescente)
            pop.sort(key=self.calcular_fitness, reverse=True)
            
            next_gen = pop[:pop_size//2] # Elitismo
            
            while len(next_gen) < pop_size:
                p1 = random.choice(next_gen)
                p2 = random.choice(next_gen)
                
                # Crossover
                cut = random.randint(0, len(p1)-1)
                child = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
                
                # Mutação
                if random.random() < 0.2:
                    i1, i2 = random.sample(range(len(child)), 2)
                    child[i1], child[i2] = child[i2], child[i1]
                
                next_gen.append(child)
            pop = next_gen
            
        pop.sort(key=self.calcular_fitness, reverse=True)
        self.my_victims = pop[0]
        print(f"{self.NAME}: Rota otimizada. Fitness={self.calcular_fitness(self.my_victims):.1f}")


    # Funcao feita com auxilio do Gemini Pro, estava com dificuldade de mapear e debuggar, foi
    def mapeamento_clusterizacao(self) -> None:
        """Une mapas, identifica vítimas (agora sem erros de ID) e clusteriza."""
        from map import Map

        # 1. Unifica o Mapa
        mapa_unico = Map()
        celulas_junt = {}
        bx, by = self.base_pos_abs 

        for mp in self.pedaco_mapa:
            for (rx, ry), dados_celula in mp.map_data.items():
                ax, ay = rx + bx, ry + by
                celulas_junt[(ax, ay)] = dados_celula

        # Garante a base como livre
        if (bx, by) not in celulas_junt:
            celulas_junt[(bx, by)] = (1.0, VS.NO_VICTIM, [VS.CLEAR]*8)

        mapa_unico.map_data = celulas_junt
        self.map = mapa_unico

        # 2. Processa Vítimas
        vitimas_junt = {}
        for d in self._victs_parts:
            vitimas_junt.update(d)
        self.victims = vitimas_junt

        self._load_pos2id()
        pos_abs = []
        victim_ids = []

        for _, (pos_local, _) in self.victims.items():
            xl, yl = pos_local
            xa, ya = self.converte_absoluto(xl, yl)

            # Validação de Segurança: O local é caminhável?
            # Se o explorador marcou como PAREDE (dificuldade alta), tentamos vizinho
            if (xa, ya) in celulas_junt and celulas_junt[(xa, ya)][0] > 50.0:
                # Procura vizinho livre
                found_better = False
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    tx, ty = xa+dx, ya+dy
                    if (tx, ty) in celulas_junt and celulas_junt[(tx, ty)][0] < 50.0:
                        xa, ya = tx, ty # Ajusta o alvo para o vizinho livre
                        found_better = True
                        break
                if not found_better:
                    print(f"AVISO: Vítima em ({xa},{ya}) parece estar numa parede.")

            # Busca o ID exato (agora que corrigimos _load_pos2id, deve bater)
            if (xa, ya) in self._pos2id:
                vid = self._pos2id[(xa, ya)]
            else:
                # Caso extremo de desalinhamento (ex: vítima em parede no arquivo)
                # Tentamos raio 1 apenas para casar o ID
                vid = -999
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if (xa+dx, ya+dy) in self._pos2id:
                            vid = self._pos2id[(xa+dx, ya+dy)]
                            break
                    if vid != -999: break
            
            pos_abs.append((xa, ya))
            victim_ids.append(vid)

        if not pos_abs:
            print("Nenhuma vítima mapeada.")
            return

        # 3. Predição e Clusterização
        # Remove IDs inválidos para a predição, mas mantém na lista
        ids_validos = [v if v >= 0 else 0 for v in victim_ids]
        sobr, tri = self.predict(ids_validos)

        # Se for -999, atribui prioridade média
        for i, v in enumerate(victim_ids):
            if v < 0:
                sobr[i] = 0.5
                tri[i] = 2

        # K-Means Espacial
        xs = np.array([x for x, _ in pos_abs], dtype=float)
        ys = np.array([y for _, y in pos_abs], dtype=float)
        
        # Normaliza para K-Means
        den_x = max(1.0, xs.max() - xs.min())
        den_y = max(1.0, ys.max() - ys.min())
        X = np.column_stack([(xs - xs.min())/den_x, (ys - ys.min())/den_y])

        k = min(3, len(pos_abs))
        os.environ["OMP_NUM_THREADS"] = "1"
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        rotulos = kmeans.fit_predict(X)

        grupos = {}
        for (x, y), vid, s, t, r in zip(pos_abs, victim_ids, sobr, tri, rotulos):
            grupos.setdefault(int(r), []).append((int(vid), int(x), int(y), float(s), int(t)))

        # 4. Distribuição
        resc_names = sorted([ag.NAME for ag in Rescuer.registry])
        self.assignments = {name: [] for name in resc_names}
        
        lookup_locs = {}
        lookup_props = {} # Para o fitness do AG

        cluster_keys = sorted(grupos.keys())
        for i, c_key in enumerate(cluster_keys):
            agente = resc_names[i % len(resc_names)]
            for v_data in grupos[c_key]:
                vid, vx, vy, vs, vt = v_data
                
                # Se ID duplicado ou inválido, gera temp único para não sobrescrever no dict
                if vid < 0 or vid in lookup_locs:
                    vid = -1000 - len(lookup_locs) 
                
                lookup_locs[vid] = (vx, vy)
                lookup_props[vid] = (vs, vt)
                self.assignments[agente].append(vid)

        print(f"Mestre: Mapa e {len(pos_abs)} vítimas distribuídas.")

        for agente in Rescuer.registry:
            agente.map = mapa_unico
            agente.victim_locs = lookup_locs
            agente.victim_props = lookup_props # Passa propriedades para o AG
            
            if agente.NAME in self.assignments:
                agente.my_victims = self.assignments[agente.NAME]
            
            agente.set_state(VS.ACTIVE)
    
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
    
    def calcular_fitness(self, sequencia):
        """
        Fitness = (10 * Soma(Pontos/Prioridade)) - (b * Custo_Bateria)
        Considera apenas vítimas alcançáveis antes da bateria acabar.
        """
        if not sequencia: return 0.0
        
        PONTOS_BASE = 100.0
        B_WEIGHT = 1.0 # Peso do custo
        
        pos_atual = (self.x, self.y)
        tempo_restante = self.get_rtime()
        
        pontos_acumulados = 0.0
        custo_acumulado = 0.0
        
        props = getattr(self, 'victim_props', {})
        
        for vid in sequencia:
            dest = self.victim_locs.get(vid)
            if not dest: continue
            
            # Custo de ida
            dist_ida = self.heuristic(pos_atual, dest)
            # Custo de volta à base (segurança)
            dist_base = self.heuristic(dest, self.base_pos_abs)
            
            custo_passo = dist_ida * self.COST_DIAG # Aproximação pessimista
            custo_retorno = dist_base * self.COST_DIAG
            
            if custo_acumulado + custo_passo + custo_retorno > tempo_restante:
                # Se não dá para ir e voltar, para a contagem aqui
                break
                
            custo_acumulado += custo_passo
            pos_atual = dest
            
            # Cálculo dos pontos
            s, t = props.get(vid, (0.5, 2)) # Padrão se não achar
            # Prioridade (Triagem): 1=Alta, 2=Media, 3=Baixa, 4=Morta?
            # Ajuste conforme seu modelo: quanto MENOR o t, MAIOR a prioridade
            prioridade = max(1, t) 
            
            ganho = (PONTOS_BASE * s) / prioridade
            pontos_acumulados += ganho
            
        return (10 * pontos_acumulados) - (B_WEIGHT * custo_acumulado)
    
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
    

    def heuristic(self, a, b):
        """Distância Manhattan ou Diagonal"""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return 1.5 * min(dx, dy) + 1.0 * abs(dx - dy)

    def aestrela(self, start, goal):
        if start not in self.map.map_data or goal not in self.map.map_data:
            # Se quiser ser muito permissivo e tentar achar caminho mesmo sem saber o goal exato:
            # return []
            pass 
        
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not frontier.empty():
            _, current = frontier.get()

            if current == goal:
                break

            # Pega dados da célula atual
            # O .get evita erro se a chave não existir
            cell_data = self.map.map_data.get(current)
            if not cell_data: continue
            
            walls = cell_data[2] # Vetor de vizinhos (walls)

            # Itera vizinhos (0 a 7)
            for i, (dx, dy) in enumerate(AbstAgent.AC_INCR.values()):
                # Verifica se a direção está bloqueada por parede
                if walls[i] != VS.WALL:
                    next_node = (current[0] + dx, current[1] + dy)
                    
                    # Verifica se o vizinho existe no mapa conhecido
                    if next_node not in self.map.map_data:
                        continue
                        
                    # Custo do movimento
                    move_cost = self.COST_DIAG if dx!=0 and dy!=0 else self.COST_LINE
                    
                    # Penalidade do terreno (se houver dificuldade de terreno)
                    n_data = self.map.map_data[next_node]
                    move_cost *= n_data[0] 

                    new_cost = cost_so_far[current] + move_cost
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + self.heuristic(next_node, goal)
                        frontier.put((priority, next_node))
                        came_from[next_node] = current

        if goal not in came_from: 
            return []
        
        path = []
        curr = goal
        while curr != start:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()
        return path

    def deliberate(self) -> bool:
        """Chamado a cada ciclo. Controla a execução."""
        
        # 0. Executa o AG uma única vez para ordenar a lista
        if not hasattr(self, 'ag_executado'):
            self.executar_ag()
            self.ag_executado = True

        # 1. Planejamento (Se não tenho rota pronta)
        if not self.plan:
            if self.current_goal is None:
                # Tenta pegar próxima vítima da lista
                while self.my_victims:
                    vid = self.my_victims[0] # Espia o primeiro, não remove ainda
                    loc = self.victim_locs.get(vid)
                    
                    if not loc:
                        self.my_victims.pop(0) # Remove se não tem local
                        continue

                    # --- CHECAGEM DE BATERIA ---
                    # 1. Custo para ir até a vítima (estimado ou real via A*)
                    rota_ida = self.aestrela((self.x, self.y), loc)

                    if not rota_ida:
                        # Debug melhorado
                        motivo = "Desconhecido"
                        if (self.x, self.y) not in self.map.map_data:
                            motivo = f"Estou em area desconhecida {(self.x, self.y)}"
                        elif loc not in self.map.map_data:
                            motivo = "Vítima em area desconhecida"
                        else:
                            motivo = "Bloqueio/Paredes no caminho"
                            
                        print(f"{self.NAME}: Sem caminho para vítima {vid}. Motivo: {motivo}. Pulando.")
                        self.my_victims.pop(0)
                        continue
                    
                    custo_ida = len(rota_ida) * self.COST_DIAG # Estimativa pessimista (tudo diagonal)
                    
                    # 2. Custo para voltar da vítima até a base
                    # Usamos heurística aqui para não rodar dois A* pesados, 
                    # mas multiplicamos por margem de segurança
                    custo_volta = self.heuristic(loc, self.base_pos_abs)
                    
                    custo_total = (custo_ida + custo_volta) * 1.5 # Margem de segurança de 50%
                    
                    if self.get_rtime() < custo_total:
                        print(f"{self.NAME}: Bateria insuficiente para ir à vítima {vid} e voltar. Abortando missão.")
                        self.my_victims = [] # Limpa lista para forçar retorno
                        break # Sai do loop para acionar retorno à base
                    
                    # Se passou na checagem, define como objetivo
                    self.my_victims.pop(0) # Remove da lista oficial
                    self.current_goal = loc
                    self.plan = rota_ida # Já calculamos, aproveita
                    print(f"{self.NAME}: Indo salvar vítima {vid}. Bateria OK.")
                    break
                
                # Se saiu do loop e não definiu current_goal, volta pra base
                if self.current_goal is None:
                    if (self.x, self.y) != self.base_pos_abs:
                        self.current_goal = self.base_pos_abs
                        print(f"{self.NAME}: Voltando para base.")
                        self.plan = self.aestrela((self.x, self.y), self.base_pos_abs)
                        if not self.plan:
                            # Se falhar o A* pra base, tenta ir vizinho a vizinho pela heuristica (desespero)
                            print(f"{self.NAME}: A* falhou para base. Tentando movimento manual.")
                            # ... (lógica de emergência opcional)
                    else:
                        print(f"{self.NAME}: Fim da execução.")
                        return False

        # 2. Execução
        if self.plan:
            prox = self.plan.pop(0)
            dx = prox[0] - self.x
            dy = prox[1] - self.y
            
            res = self.walk(dx, dy)
            
            if res == VS.BUMPED:
                print(f"{self.NAME}: Colisão! Recalculando.")
                self.plan = []
            elif res == VS.EXECUTED:
                self.x += dx
                self.y += dy
            
            if (self.x, self.y) == self.current_goal:
                if self.current_goal != self.base_pos_abs:
                    print(f"{self.NAME}: Salvando vítima...")
                    self.first_aid()
                self.current_goal = None

        return True