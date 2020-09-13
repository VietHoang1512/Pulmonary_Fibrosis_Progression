import tensorflow as tf
import efficientnet.tfkeras as efn


def get_efficientnet(model, shape):
    models_dict = {
        'b0': efn.EfficientNetB0(input_shape=shape, weights=None, include_top=False),
        'b1': efn.EfficientNetB1(input_shape=shape, weights=None, include_top=False),
        'b2': efn.EfficientNetB2(input_shape=shape, weights=None, include_top=False),
        'b3': efn.EfficientNetB3(input_shape=shape, weights=None, include_top=False),
        'b4': efn.EfficientNetB4(input_shape=shape, weights=None, include_top=False),
        'b5': efn.EfficientNetB5(input_shape=shape, weights=None, include_top=False),
        'b6': efn.EfficientNetB6(input_shape=shape, weights=None, include_top=False),
        'b7': efn.EfficientNetB7(input_shape=shape, weights=None, include_top=False)
    }
    return models_dict[model]


def build_model(shape=(512, 512, 1), model_class=None):
    inp = tf.keras.layers.Input(shape=shape)
    base = get_efficientnet(model_class, shape)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    inp2 = tf.keras.layers.Input(shape=(4,))
    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)
    x = tf.keras.layers.Concatenate()([x, x2])
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model([inp, inp2], x)
    return model
