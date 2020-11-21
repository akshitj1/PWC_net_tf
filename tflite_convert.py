from model import build_model, pyramid_compatible_shape
# import tensorflow as tf

if __name__ == "__main__":
    BATCH_SIZE = 1
    NUM_PYRAMID_LEVELS = 8
    PREDICT_LEVEL = 2
    DATASET_SHAPE = (436, 1024) # h, w
    MODEL_IN_SHAPE = pyramid_compatible_shape(DATASET_SHAPE, NUM_PYRAMID_LEVELS)

    model = build_model(MODEL_IN_SHAPE, NUM_PYRAMID_LEVELS, PREDICT_LEVEL, True)
    status = model.load_weights('pwc_net_ckpts/trained_ckpt/ckpt')
    print("weights loaded with status: {}".format(status))

    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    lite_model = converter.convert()
    print("model converted successfully")

    lite_model_file = "pwc_net_ckpts/pwc_net.tflite"
    open(lite_model_file, "wb").write(lite_model)
    print("model written to: {}".format(lite_model_file))