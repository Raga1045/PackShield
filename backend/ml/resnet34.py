"""Model architectures for DefektGuard.

Includes:
1. Custom ResNet-34 architecture with BasicBlock residual units [3, 4, 6, 3]
2. Standard Baseline CNN (4-layer convolutional network) for ablation/comparison
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def basic_block(input_tensor, filters, stride=1, weight_decay=1e-4, name=None):
    """ResNet BasicBlock consisting of 2 3x3 convolutions with residual connection."""
    name_prefix = name or "res_block"
    
    # First 3x3 conv
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=f"{name_prefix}_conv1",
    )(input_tensor)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)
    
    # Second 3x3 conv
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=f"{name_prefix}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    
    # Shortcut connection (projection if dimension changes, identity otherwise)
    shortcut = input_tensor
    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            name=f"{name_prefix}_shortcut_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)
    
    x = layers.add([x, shortcut], name=f"{name_prefix}_add")
    x = layers.ReLU(name=f"{name_prefix}_out")(x)
    return x


def resnet_stage(input_tensor, filters, num_blocks, stride=1, weight_decay=1e-4, stage_num=1):
    """Builds a ResNet stage composed of multiple BasicBlocks."""
    x = basic_block(
        input_tensor,
        filters=filters,
        stride=stride,
        weight_decay=weight_decay,
        name=f"stage{stage_num}_block1",
    )
    for i in range(1, num_blocks):
        x = basic_block(
            x,
            filters=filters,
            stride=1,
            weight_decay=weight_decay,
            name=f"stage{stage_num}_block{i+1}",
        )
    return x


def build_resnet34(
    input_shape=(256, 256, 3),
    num_classes=2,
    weight_decay=1e-4,
    dropout_rate=0.2,
    name="ResNet34",
):
    """Constructs a custom ResNet-34 architecture [3, 4, 6, 3] in Keras."""
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Stem: 7x7 Conv -> BN -> ReLU -> 3x3 MaxPool
    x = layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="stem_pool")(x)
    
    # 4 Residual Stages: [3, 4, 6, 3]
    x = resnet_stage(x, filters=64, num_blocks=3, stride=1, weight_decay=weight_decay, stage_num=1)
    x = resnet_stage(x, filters=128, num_blocks=4, stride=2, weight_decay=weight_decay, stage_num=2)
    x = resnet_stage(x, filters=256, num_blocks=6, stride=2, weight_decay=weight_decay, stage_num=3)
    x = resnet_stage(x, filters=512, num_blocks=3, stride=2, weight_decay=weight_decay, stage_num=4)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(weight_decay),
        name="classification_output",
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def build_baseline_cnn(
    input_shape=(256, 256, 3),
    num_classes=2,
    dropout_rate=0.3,
    name="BaselineCNN",
):
    """Constructs a standard 4-layer Convolutional Neural Network baseline."""
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    
    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def build_transfer_model(
    input_shape=(256, 256, 3),
    num_classes=2,
    dropout_rate=0.2,
    name="DefektGuardTransferModel",
):
    """Constructs a high-accuracy transfer learning architecture with ImageNet backbone."""
    from pathlib import Path
    local_weights = (
        Path(__file__).resolve().parent.parent.parent
        / ".keras"
        / "models"
        / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    )
    weights_arg = str(local_weights) if local_weights.exists() else "imagenet"

    inputs = layers.Input(shape=input_shape, name="image_input")
    scaled = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_rescaling")(inputs)
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=scaled,
        weights=weights_arg,
        include_top=False,
    )
    base_model.trainable = False

    x = base_model.output
    avg_pool = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    x = layers.Concatenate(name="concat_pool")([avg_pool, max_pool])
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    x = layers.Dense(256, activation="relu", name="dense_features_1")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="feature_dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_features_2")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classification_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


