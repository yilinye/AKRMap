import json
import datetime

import torch

from akrmap.akr_model import NeuralMapper
from akrmap.akr_training import fit_akrmap_model, MapLoss


def train_parametric_akr_map_model(points_ds, input_dimens, config, scores):
    net = NeuralMapper
    ffnn = net(dim_input=input_dimens).to(torch.device(config.dev))
    maploss = MapLoss()
    opt = torch.optim.Adam([{"params": ffnn.parameters(), "params1": maploss.parameters()}], **config.optimization_conf)

    report_config = json.dumps(
        {"device": config.dev,
         "seed": config.seed,
         "optimization": config.optimization_conf,
         "training": config.training_params})

    start = datetime.datetime.now()

    model_path, a1, a2, b=fit_akrmap_model(ffnn,
                    points_ds,
                    opt,
                    **config.training_params,
                    epochs_to_save_after=config.epochs_to_save_after,
                    dev=config.dev,
                    save_dir_path=config.save_dir_path,
                    configuration_report=report_config,
                    maploss=maploss,
                    scores=scores,
                    config=config,
                    )

    fin = datetime.datetime.now()
    print("Training time:", fin - start, flush=True)

    return model_path, a1, a2, b