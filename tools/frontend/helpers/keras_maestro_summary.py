def summary(model):
    mae_summary = {}
    
    for i in range(len(model.layers)):
        cur_config = model.layers[i].get_config()
        cur_config['type'] = model.layers[i].__class__.__name__

        is_conv2d = cur_config['type'] == 'Conv2D'
        is_dwconv2d = cur_config['type'] == 'DepthwiseConv2D'
        is_dense = cur_config['type'] == 'Dense'

        if is_conv2d:
            cur_config['type'] = 'CONV'
        if is_dwconv2d:
            cur_config['type'] = 'DSCONV'

        if(is_conv2d or is_dwconv2d or is_dense):
            cur_ic = []
            cur_ic.append(None)
            if(is_conv2d):
                cur_ic.append(model.layers[i].output_shape[3])
            elif(is_dwconv2d):
                cur_ic.append(1)
            elif(is_dense):
                cur_ic.append(model.layers[i].output_shape[1])

            if(is_dense):
                cur_ic.append(model.layers[i].input_shape[1])
                for _ in range(4):
                    cur_ic.append(1)
            else:
                cur_ic.append(model.layers[i].input_shape[3])
                cur_ic.append(model.layers[i].kernel_size[0])
                cur_ic.append(model.layers[i].kernel_size[1])
                cur_ic.append(model.layers[i].input_shape[1])
                cur_ic.append(model.layers[i].input_shape[2])

            cur_config['dimension_ic'] = cur_ic
        mae_summary[model.layers[i].get_config()['name']] = cur_config

    return mae_summary
