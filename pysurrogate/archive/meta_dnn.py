import numpy
import tensorflow as tf

from pysao.metamodels.metamodel import Metamodel


class DNNMetamodel(Metamodel):
    def __init__(self, n_neural_nets=1):
        Metamodel.__init__(self)
        self.n_neural_nets = n_neural_nets
        self.estimators = None

    def _predict(self, X):

        vals = []
        for estimator in self.estimators:
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": X},
                num_epochs=1,
                shuffle=False)

            val = estimator.predict(input_fn=predict_input_fn)
            val = [e for e in list(val)]
            vals.append(numpy.array([e['predictions'] for e in val]).T[0])

        vals = numpy.array(vals)

        return numpy.median(vals, axis=0), numpy.std(vals, axis=0)

    def _fit(self, X, F, data):

        self.estimators = []
        feature_columns = [tf.feature_column.numeric_column("x", shape=[X.shape[1]])]
        input_fn = tf.estimator.inputs.numpy_input_fn({"x": X}, F, batch_size=4, num_epochs=None, shuffle=True)

        for _ in range(self.n_neural_nets):
            estimator = tf.estimator.DNNRegressor([1024, 512, 256], feature_columns=feature_columns)
            estimator = estimator.train(input_fn=input_fn, steps=100000)
            self.estimators.append(estimator)

    @staticmethod
    def get_params():
        return [{}]
