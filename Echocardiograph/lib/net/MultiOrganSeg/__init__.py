from .MyNetWork import get_network as DefultNetwork
from .VnetOri import get_network as VnetOri

_network_factory = {
    'Default': DefultNetwork,
    'Vnet_Ori': VnetOri,
}


def get_network(cfg, is_train):
    arch = cfg.network
    get_model = _network_factory[arch]  # -- .Net.getnetwork()
    network = get_model(cfg, is_train)

    return network
