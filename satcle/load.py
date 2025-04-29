from main_satcle import *

def get_satcle(ckpt_path, device, return_all=False):
    ckpt = torch.load(ckpt_path,map_location=device)
    if 'eval_downstream' in ckpt['hyper_parameters']:
        ckpt['hyper_parameters'].pop('eval_downstream')
    if 'air_temp_data_path' in ckpt['hyper_parameters']:
        ckpt['hyper_parameters'].pop('air_temp_data_path')
    if 'election_data_path' in ckpt['hyper_parameters']:
        ckpt['hyper_parameters'].pop('election_data_path')
    # print(ckpt['hyper_parameters'])
    lightning_model = SatCLELightningModule(**ckpt['hyper_parameters']).to(device)

    lightning_model.load_state_dict(ckpt['state_dict'])
    lightning_model.eval()

    geo_model = lightning_model.model

    if return_all:
        return geo_model
    else:
        return geo_model.location