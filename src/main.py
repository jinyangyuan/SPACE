from engine.utils import get_config
from engine.train import train
from engine.eval import eval
from engine.show import show


if __name__ == '__main__':

    task_dict = {
        'train': train,
        'eval': eval,
        'show': show,
    }
    cfg, task, data_loaders, config = get_config()
    assert task in task_dict
    task_dict[task](cfg, data_loaders, config)


