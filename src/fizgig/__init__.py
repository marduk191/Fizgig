# Fizgig - Independent LoRA Training Framework for Klein 9B
__version__ = "0.1.0"

from fizgig.utils.container_ram import install as _container_ram
_container_ram()     # a pod's RAM checks read its own limit, not the host's (Linux containers only)
