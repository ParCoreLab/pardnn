def model_fn(features, labels, mode):
    X = features['x']
    
    # Define weights initializer
    kernel_initializer = tf.initializers.truncated_normal(stddev=0.1)
 
    # Define network  
    hidden = tf.layers.dense(
        X, 2, activation=tf.nn.relu, 
        kernel_initializer=kernel_initializer)
    
    logits = tf.layers.dense(
        hidden, 3, 
        kernel_initializer=kernel_initializer)
    # Define dictionary with predicted class 
    # and probabilities for each class
    predictions = {
        'class': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits)
    }
    # If we are on prediction mode, 
    # we return estimator specification
    # with predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)
    
    onehot_labels = tf.one_hot(labels, depth=3)       
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)
    # If we are on training mode, 
    # we need to do few things.
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        
        # Define training operation (minimize loss function)        
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
        # Return estimator specification 
        # with loss and training operations  
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)
    
    # If we are on evaluation mode,
    # we need to define metrics operations
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels, predictions['class'])
    }
    # Return estimator specification 
    # with loss and metrics operations   
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)