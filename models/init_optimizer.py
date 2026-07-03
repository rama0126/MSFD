import torch
from torch.optim import Adam, SGD, AdamW, Adadelta, Adagrad, Adamax, RMSprop, Rprop
from torch.optim import lr_scheduler


def init_optimizer(model, lr, optimizer_, scheduler_, epochs, args):

    if optimizer_ == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)    
    elif optimizer_ == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif optimizer_ == 'sgd-momentum':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    elif optimizer_ == 'adadelta':
        optimizer = Adadelta(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif optimizer_ == 'adagrad':
        optimizer = Adagrad(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif optimizer_ == 'adamax':
        optimizer = Adamax(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif optimizer_ == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif optimizer_ == 'rprop':
        optimizer = Rprop(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif optimizer_ == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer')
    
    if scheduler_ == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_sche_step_size, gamma=args.lr_sche_gamma)
    elif scheduler_ == 'OneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=lr, 
        steps_per_epoch=1, 
        pct_start=0.2,
        epochs=epochs)
    else:
        scheduler = None
    return optimizer, scheduler
