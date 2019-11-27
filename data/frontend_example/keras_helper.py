import tensorflow.keras.applications as keras_models

'''
Available model from tf 2.0:
Xception
VGG16
VGG19
ResNet, ResNetV2
InceptionV3
InceptionResNetV2
MobileNet
MobileNetV2
DenseNet
NASNet
'''

def get_model(model_name, input_shape):
    keras_model_name = ''
    if(model_name == 'xception'):
        keras_model_name = 'Xception'

    elif(model_name == 'vgg16'):
        keras_model_name = 'VGG16'
    elif(model_name == 'vgg19'):
        keras_model_name = 'VGG19'

    elif(model_name == 'resnet50'):
        keras_model_name = 'ResNet50'
    elif(model_name == 'resnet101'):
        keras_model_name = 'ResNet101'
    elif(model_name == 'resnet152'):
        keras_model_name = 'ResNet152'
    elif(model_name == 'resnet50_v2'):
        keras_model_name = 'ResNet50V2'
    elif(model_name == 'resnet101_v2'):
        keras_model_name = 'ResNet101V2'
    elif(model_name == 'resnet152_v2'):
        keras_model_name = 'ResNet152V2'

    elif(model_name == 'inception_v3'):
        keras_model_name = 'InceptionV3'
    elif(model_name == 'inception_resnet_v2'):
        keras_model_name = 'InceptionResNetV2'

    elif(model_name == 'mobilenet'):
        keras_model_name = 'MobileNet'
    elif(model_name == 'mobilenet_v2'):
        keras_model_name = 'MobileNetV2'

    elif(model_name == 'densenet121'):
        keras_model_name = 'DenseNet121'
    elif(model_name == 'densenet169'):
        keras_model_name = 'DenseNet169'
    elif(model_name == 'densenet201'):
        keras_model_name = 'DenseNet201'

    elif(model_name == 'nasnet_large'):
        keras_model_name = 'NASNetLarge'
    elif(model_name == 'nasnet_mobile'):
        keras_model_name = 'NASNetMobile'

    else:
        raise NotImplementedError('Not supported model')
    keras_model = getattr(keras_models, keras_model_name)(weights=None, include_top=True, input_shape=input_shape)
    print('Get the keras model: ' + keras_model_name)

    return keras_model
