import tensorflow as tf
from tfrecord_utils import parse_function

def input_fn(filenames, buffer_size, batch_size):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parse_function).shuffle(buffer_size).batch(batch_size)
    return dataset.repeat()

def make_model():
    print('salam')_
    NUM_CLASSES = 10
    IMAGE_HEIGHT, IMAGE_WIDTH = 32, 32
    NUM_CHANNELS = 3
    
    return tf.keras.Sequential([
        tf.keras.applications.DenseNet121(include_top=False,
                                        weights='imagenet',
                                        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS),
                                        pooling='max'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
                                ])

def model_fn(features, labels, mode):
    model = make_model()

    optimizer = tf.compat.v1.train.AdamOptimizer()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    predictions = model(features, training=training)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

#     reg_losses = model.get_losses_for(None) + model.get_losses_for(features)
    total_loss = loss_fn(labels, predictions)# + tf.math.add_n(reg_losses)

    accuracy = tf.compat.v1.metrics.accuracy(labels=labels,
                                            predictions=tf.math.argmax(predictions, axis=1),
                                            name='acc_op')

    update_ops = model.get_updates_for(None) + model.get_updates_for(features)
    minimize_op = optimizer.minimize(
                                    total_loss,
                                    var_list=model.trainable_variables,
                                    global_step=tf.compat.v1.train.get_or_create_global_step())
    train_op = tf.group(minimize_op, update_ops)

    return tf.estimator.EstimatorSpec(
                                    mode=mode,
                                    predictions=predictions,
                                    loss=total_loss,
                                    train_op=train_op, eval_metric_ops={'accuracy': accuracy})
