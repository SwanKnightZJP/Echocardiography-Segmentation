from .MyNetWork import get_network as DefultNetwork
from .VnetOri import get_network as VnetOri
from .Vnet8x import get_network as Vnet8x
from .Vnet8xD import get_network as Vnet8xD

_network_factory = {
    'Default': DefultNetwork,
    'Vnet_Ori': VnetOri,
    'Vnet8x': Vnet8x,
    'Vnet8xD': Vnet8xD,
}


def get_network(cfg, is_train):
    arch = cfg.network
    # num_classes = cfg.classes
    get_model = _network_factory[arch]  # -- .Net.getnetwork()
    # dropout_rate = cfg.dpr
    # network = get_model(num_classes, is_train, dropout_rate)
    network = get_model(cfg, is_train)

    return network
