import torch


def train_embedding(inputs, model: torch.nn, args):
    input_height, input_width = inputs[0][1].shape[0], inputs[0][1].shape[1]

    train_loss_log = []
    test_loss_log = []

    model.tranin()
    optimizer = torch.optim.Adam(model.parameter)
    for epoch in range(args.epochs):
