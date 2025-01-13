import torch
from weaver.nn.model.ParticleNet import ParticleNetTagger


def get_model(data_config, **kwargs):
    ec_k = kwargs.get('ec_k', 16)
    ec_c1 = kwargs.get('ec_c1', 64)
    ec_c2 = kwargs.get('ec_c2', 128)
    ec_c3 = kwargs.get('ec_c3', 256)
    fc_c, fc_p = kwargs.get('fc_c', 256), kwargs.get('fc_p', 0.1)
    conv_params = [
        (ec_k, (ec_c1, ec_c1, ec_c1)),
        (ec_k, (ec_c2, ec_c2, ec_c2)),
        (ec_k, (ec_c3, ec_c3, ec_c3)),
    ]
    fc_params = [(fc_c, fc_p)]

    use_fusion=True

    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = len(data_config.label_value)
    model = ParticleNetTagger(pf_features_dims, sv_features_dims, num_classes,
                        conv_params, fc_params,
                        use_fusion=use_fusion,
                        use_fts_bn=kwargs.get('use_fts_bn', False),
                        use_counts=kwargs.get('use_counts', True),
            			pf_input_dropout=kwargs.get('pf_input_dropout', None),
                        sv_input_dropout=kwargs.get('sv_input_dropout', None),
                        for_inference=kwargs.get('for_inference', False)
                        )
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    #return torch.nn.BCELoss()
    return torch.nn.CrossEntropyLoss()
