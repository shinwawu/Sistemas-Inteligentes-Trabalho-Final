import os

# importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer
import warnings

warnings.filterwarnings("ignore")


def resumo_exploracao(rescuers):
    """
    mostra as informacoes da exploracao.
    """
    mestres = [r for r in rescuers if getattr(r, "is_master", False)]
    if not mestres:
        print("socorrista mestre nao encontrado.\n")
        return

    mestre = mestres[0]
    vitimas_por_explorador = [len(v) for v in mestre._victs_parts]

    todas_vitimas = [v for parte in mestre._victs_parts for v in parte.values()]
    vitimas_unicas = {pos_vs[0] for pos_vs in todas_vitimas}
    total_unicas = len(vitimas_unicas)

    soma_total = sum(vitimas_por_explorador)
    sobreposicao = ((soma_total / total_unicas) - 1) if total_unicas > 0 else 0.0

    ve1, ve2, ve3 = (vitimas_por_explorador + [0, 0, 0])[:3]

    print(
        f"resultados: Ve1={ve1} , Ve2={ve2} , Ve3={ve3} , "
        f"Ve={total_unicas} , sobreposicao={sobreposicao:.3f}"
    )


def _run_env_once(vict_folder, env_folder, config_ag_folder):
    """
    inicia as variaveis ambiente,vitimas, e os agentes. roda o simulador também.
    foi utilizado chatgpt para debugar abaixo, que a instância não estava sendo limpa
    """
    # limpa a instancia apos a execucao tml=1000
    Rescuer.registry.clear()

    # Instantiate the environment
    env = Env(vict_folder, env_folder)

    # Instantiate agents rescuer and explorer
    resc_config_root = os.path.join(os.getcwd(), "config_ag_resc")

    rescuers = []
    for i in range(1, 4):
        resc_file = os.path.join(resc_config_root, f"rescuer_{i}.txt")
        is_master = i == 1
        resc = Rescuer(
            env,
            resc_file,
            is_master=is_master,
            total_explorers=3,
            env_victims_path="env_victims.txt",
            data_csv="data.csv",
            modelo_sobr="modelo_sobrevivencia.pkl",
            modelo_tria="modelo_triagem.pkl",
            env_config_path="env_config.txt",
        )
        rescuers.append(resc)

    def entregamapa(explorer_name, local_map, victims_dict):
        """
        entrega mapa aos socorristas
        """
        for r in rescuers:
            r.recebe_mapa(explorer_name, local_map, victims_dict)

    for i in range(1, 4):
        explorer_file = os.path.join(config_ag_folder, f"explorer_{i}.txt")
        # Explorer needs to know rescuer to send the map
        # that's why rescuer is instatiated before
        Explorer(env, explorer_file, entregamapa)

    env.run()
    resumo_exploracao(rescuers)


def main(vict_folder, env_folder, config_ag_folder):
    """
    inicia a funcao de iniciar variaveis de ambiente
    """
    _run_env_once(vict_folder, env_folder, config_ag_folder)


if __name__ == "__main__":
    # dataset com sinais vitais das vitimas
    vict_folder = os.getcwd()

    # dataset do ambiente (paredes, posicao das vitimas)
    env_folder = os.getcwd()

    print("\n==============================")
    print("==== INÍCIO SMA | TLIM=8000 ====")
    print("==============================")
    config_ag_folder = os.path.join(os.getcwd(), "config_ag_1000")
    main(vict_folder, env_folder, config_ag_folder)
