from .space.space import Space

__all__ = ['get_model']

def get_model(cfg, config):
    """
    Also handles loading checkpoints, data parallel and so on
    :param cfg:
    :return:
    """
    
    model = None
    if cfg.model == 'SPACE':
        model = Space(config)
        
    return model
