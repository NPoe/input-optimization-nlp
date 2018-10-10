import numpy as np
np.random.seed(123)

import sys
import os

import argparse
import _pickle

from keras.layers import Layer, add, Embedding, Dense, Lambda, Input, TimeDistributed, Activation
from keras.models import Sequential, Model, load_model
from keras.callbacks import Callback, CSVLogger, EarlyStopping
import keras.backend as K

from progressbar import ProgressBar
from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import linkage

import paths

def maximizer_loss(y_true, y_pred):
    return -K.mean(y_pred)

def report_loss(y_true, y_pred):
    return K.mean(y_pred)

class KeepBestModelCallback(Callback):
    def __init__(self, monitor, mode = "auto", verbose = 1):
        self.monitor = monitor
        self.verbose = verbose

        if mode not in ["auto", "min", "max"]:
            raise Exception("Mode must be in 'auto', 'min', 'max'")

        if mode == "auto":    
            if "acc" in self.monitor or "fmeasure" in self.monitor:
                        mode = "max"
            elif "loss" in self.monitor:
                mode = "min"
            else:
                raise Exception("Cannot infer mode from", self.monitor)
    
        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf

    def on_train_begin(self, logs = None):
        self.weights = self.model.get_weights()
        
    def on_epoch_end(self, epoch, logs = None):
        current = logs.get(self.monitor)
        if current is None:
            raise Exception("Cannot monitor", self.monitor, "as it is not present in the model logs")
            
        if self.monitor_op(current, self.best):
            if self.verbose > 0:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f\n' % (epoch+1, self.monitor, self.best, current))
            self.weights = self.model.get_weights()
            self.best = current
        
    def on_train_end(self, logs = None):
        self.model.set_weights(self.weights)


        
class BestStringCallback(Callback):
    def __init__(self, optimizer, generator):
        self.optimizer = optimizer
        self.rev_dictionary = optimizer.rev_dictionary
        self.model = optimizer.model
        self.generator = generator
        self.name_to_index = {name: i for i, name in enumerate(self.model.output_names)}
    
    def on_epoch_end(self, epoch, logs):
        tmp = self.model.predict_on_batch(next(self.generator)[0])
        if "t" in self.name_to_index:
            print("Temperature:", tmp[self.name_to_index["t"]][0].squeeze())
        
        statistics = []
        for name in sorted(list(self.name_to_index.keys())):
            if name in ("o_softmax", "o_logits", "o_argmax", "o_gumbel"):
                outputs = tmp[self.name_to_index[name]]
                if name == "o_gumbel":
                    outputs = outputs[:5]
                else:
                    outputs = outputs[:1]

                for output in outputs:
                    statistics.append((name, output.argmax(axis = -1), output.max(axis = -1)))

            elif name == "o_emb":
                output = tmp[self.name_to_index[name]][0]
                similarities = cosine_similarity(output, self.optimizer.weights["emb"][0])
                statistics.append((name, similarities.argmax(axis = -1), similarities.max(axis = -1)))
        
        for name, words, stats in statistics:
            print(name, end = ": ")
            print(" ".join(self.rev_dictionary[w] + " (" + str(round(s, 4)) + ")" for w, s in zip(words, stats)))
    
    
class SnapToClosestLayer(Layer):
    def __init__(self, reference, mode, **kwargs):
        super(SnapToClosestLayer, self).__init__(**kwargs)
        self.reference = reference
        if mode == "min":
            self.op = K.argmin
        elif mode == "max":
            self.op = K.argmax
        else:
            raise Exception()

    def call(self, inputs):
        positions = self.op(inputs, axis = -1)
        return K.gather(self.reference, positions)
    
class PairwiseCosinesLayer(Layer):
    def __init__(self, reference, **kwargs):
        super(PairwiseCosinesLayer, self).__init__(**kwargs)
        self.reference_norm = K.expand_dims(K.l2_normalize(reference, axis = -1), -3) # (..., 1, vocab, emb)
    
    def call(self, inputs):
        inputs += K.epsilon()
        inputs_norm = K.expand_dims(K.l2_normalize(inputs, axis = -1), -2) # (..., length, 1, emb)
        return K.sum(inputs_norm * self.reference_norm, axis = -1) # (..., length, vocab)

class NameLayer(Layer):
    def __init__(self, name, **kwargs):
        super(NameLayer, self).__init__(name = name, **kwargs)


class GumbelSoftmaxLayer(Layer):
    def __init__(self, temperature = None, **kwargs):
        super(GumbelSoftmaxLayer, self).__init__(**kwargs)
        self.temperature = temperature
    
    def compute_output_shape(self, input_shape):
        if self.temperature is None:
            if not (isinstance(input_shape, list) and len(input_shape) >= 2):
                raise Exception("If you do not specify a temperature parameter, it is expected as last input")
            input_shape = input_shape[:-1]
        
        if len(input_shape) == 1:
            return input_shape[0]
        return input_shape
    
    def call(self, inputs):
        if self.temperature is None:
            if not (isinstance(inputs, list) and len(inputs) >= 2):
                raise Exception("If you do not specify a temperature parameter, it is expected as last input")
            
            temperature = inputs[-1]
            inputs = inputs[:-1]
        else:
            temperature = self.temperature

        if not hasattr(inputs, "__len__"):
            inputs = [inputs]

        outputs = []

        for x in inputs:
            gumbel_noise = -K.log(-K.log(K.random_uniform(shape = K.shape(x), minval = 0.0, maxval = 1.0)))
            
            for _ in range(K.ndim(x) - K.ndim(temperature)):
                temperature = K.expand_dims(temperature, -1)

            outputs.append(K.softmax((K.log(x + K.epsilon()) + gumbel_noise) / temperature, -1))

        if len(outputs) == 1:
            return outputs[0]
        return outputs


class NeuronSelector(Layer):
    def __init__(self, indices, **kwargs):
        super(NeuronSelector, self).__init__(**kwargs)
        self.indices = indices

    def call(self, inputs):
        return K.stack([inputs[:,i] for i in self.indices], axis = -1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (len(self.indices),)


class Argmax(Layer):
    def call(self, inputs):
        return K.one_hot(K.argmax(inputs, axis = -1), K.shape(inputs)[-1])


class _Optimizer:
    def __init__(self, indices, model, weightdir, csvdir, textdir, dictionary, modality, length, target_layer, verbose):
        
        self.with_projection = True
        if target_layer == "hidden":
            self.with_projection = False

        self.original_model = model
        self.dictionary = dictionary
        self.modality = modality
        self.length = length
        self.verbose = verbose
        self.indices = indices
        self.textdir = textdir
        
        self.csvpath = os.path.join(csvdir, "{}_{}_{}_{}.csv".format(self.name, "_".join(str(i) for i in indices), modality, target_layer))
        self.weightpath = os.path.join(csvdir, "{}_{}_{}_{}.pkl".format(self.name, "_".join(str(i) for i in indices), modality, target_layer))
        self._check_indices()
    
    def _check_indices(self):
        if not hasattr(self.indices, "__len__") or hasattr(self.indices[0], "__len__"):
            raise Exception("All optimizers except for OptimizerSearch require a simple list of indices: [1,2,6, ...]")
        
    def build(self):
        self._prep()
        self._build()
        self._assemble()
    
    def _prep(self):
        if isinstance(self.original_model, str):
            self.original_model = load_model(self.original_model)
        
        if isinstance(self.dictionary, str):
            with open(self.dictionary, "rb") as handle:
                self.dictionary = _pickle.load(handle)
        
        self.rev_dictionary = {self.dictionary[key]: key for key in self.dictionary}
        
        orig_emb = self.original_model.get_layer("emb")
        orig_enc = self.original_model.get_layer("enc_" + self.modality)
    
        self.weights = {}
        self.weights["emb"] = orig_emb.get_weights()
        self.weights["enc"] = orig_enc.get_weights()
        
        enc_config = orig_enc.get_config()
        enc_config["return_sequences"] = False
        emb_config = orig_emb.get_config()
        emb_config["mask_zero"] = False

        self.configs = {}
        self.configs["emb"] = emb_config
        self.configs["enc"] = enc_config
        self.configs["enc"]["return_sequences"] = False

        if self.with_projection:
            orig_proj = self.original_model.get_layer(self.modality)
            self.weights["proj"] = orig_proj.get_weights()

            if isinstance(orig_proj, TimeDistributed):
                self.configs["proj"] = orig_proj.layer.get_config()
            else:
                self.configs["proj"] = orig_proj.get_config()
            
            self.configs["proj"]["activation"] = "linear"
            
            if self.modality == "T":
                orig_proj = Dense(**self.configs["proj"], weights = self.weights["proj"])

        for key in self.configs:
            self.configs[key]["trainable"] = False
            del self.configs[key]["name"]
    
        if self.with_projection:
            self.projection = Dense(**self.configs["proj"], weights = self.weights["proj"], name = "proj_exp")
                
        self.encoder = orig_enc.__class__(**self.configs["enc"], weights = self.weights["enc"], name = "enc_exp")
        self.orig_model = Sequential([orig_emb, orig_enc])
        
        if self.with_projection:
            self.orig_model.add(orig_proj)
        


class OptimizerSearch(_Optimizer):
    name = "search"

    def __init__(self, batchsize = 512, **kwargs):
        super(OptimizerSearch, self).__init__(**kwargs)
        self.batchsize = batchsize
    
    def _build(self):
        input = Input((self.length,))
        embedded = self.embedding(input)
        encoded = self.encoder(embedded)
        if self.with_projection:
            encoded = self.projection(encoded)

        self.model = Model([input], [encoded])
                
    def _assemble(self):
        if self.verbose > 1:
            self.model.summary()

    def _check_indices(self):
        if not hasattr(self.indices, "__len__") or not hasattr(self.indices[0], "__len__"):
            raise Exception("OptimizerSearch requires a nested list of indices: [[0,1], [1,2,6], ...]")

    def _prep(self):
        super(OptimizerSearch, self)._prep()
        
        self.configs["emb"]["mask_zero"] = True
        self.embedding = Embedding(**self.configs["emb"], weights = self.weights["emb"], name = "emb_exp")

        self.ngrams = set()
        for dataset in ("train", "test", "val"):
            with open(os.path.join(self.textdir, dataset + ".txt")) as handle:
                for line in handle:
                    words = line.strip().split()
                    words = [self.dictionary.get(w, self.dictionary["<UNK>"]) for w in words]
                    
                    if len(words) < self.length:
                        words = [0] * (self.length - len(words)) + words

                    for i in range(0, len(words) - self.length + 1):
                        self.ngrams.add(tuple(words[i:i+self.length]))

        self.ngrams = np.array(list(self.ngrams))


    def train(self):
        self.best_ngrams = [None for _ in range(len(self.indices))]
        self.best_scores = [-np.inf for _ in range(len(self.indices))]

        batchrange = range(0, self.ngrams.shape[0], self.batchsize)
        
        if self.verbose > 1:
            bar = ProgressBar()
            batchrange = bar(batchrange)

        for i in batchrange:
            batch = self.ngrams[i:min(self.ngrams.shape[0], i + self.batchsize)]
            out = self.model.predict_on_batch(batch)
            
            for j, index_group in enumerate(self.indices):
                scores = np.mean([out[:,i] for i in index_group], axis = 0)
                if self.best_scores[j] < scores.max():
                    self.best_scores[j] = scores.max()
                    self.best_ngrams[j] = self.ngrams[scores.argmax() + i]
    
    def get_best_ngrams(self):
        return [" ".join(self.rev_dictionary[w] for w in ngram) for ngram in self.best_ngrams]
    
    def score_best_ngrams(self):
        return self.best_scores    
    

class _OptimizerParametrical(_Optimizer):

    def __init__(self, batchsize = 1, **kwargs):
        super(_OptimizerParametrical, self).__init__(**kwargs)
        self.calc_hard = True
        self.batchsize = batchsize
    
    def _base_generator(self):
        dummy_in = np.tile(np.arange(self.length), (self.batchsize, 1))
        while True:
            yield({"i_dummy": dummy_in}, [])

    def _get_generator(self):
        base_generator = self._base_generator()
        while True:
            yield(next(base_generator))
    
    def _temperature_generator(self):
        if self.annealing_factor > 1:
            start_temperature = self.min_temperature
            end_temperature = self.max_temperature
            monitor_op = np.greater
        else:
            start_temperature = self.max_temperature
            end_temperature = self.min_temperature
            monitor_op = np.less

        temperature = start_temperature
        while True:
            yield(temperature)
            temperature *= self.annealing_factor
            if monitor_op(temperature, end_temperature):
                temperature = end_temperature
            
    
    def _prep(self):
        super(_OptimizerParametrical, self)._prep()
        self.dummy_input = Input((self.length,), name = "i_dummy")
        self.inputs = [self.dummy_input]
        self.outputs = []
        self.dummy_target = K.constant([[0]])
        self.selector = NeuronSelector(self.indices, name = "selector")

    def _assemble(self):        
        self.losses = {}
        loss_weights = {}
        target_tensors = {}

        for tensor in self.outputs:
            name = tensor._keras_history[0].name
            
            if name.startswith("o_"): continue
            
            target_tensors[name] = K.constant([[0]])

            if name == "L":
                self.losses[name] = maximizer_loss
                loss_weights[name] = 1
            else:
                self.losses[name] = report_loss
                loss_weights[name] = 0
        
        self.model = Model(self.inputs, self.outputs)
        self.model.compile(optimizer = "adam", loss = self.losses, 
            loss_weights = loss_weights, target_tensors = target_tensors)
    
        assert len(self.model.trainable_weights) == 1

        if self.verbose > 1:
            self.model.summary()

    def train(self):
        generator = self._get_generator()
        
        if self.calc_hard:
            monitor = "h_loss"
        else:
            monitor = "s_loss"

        callbacks = []
        callbacks.append(CSVLogger(self.csvpath))
        callbacks.append(EarlyStopping(monitor = monitor, mode = "max", patience = self.patience, verbose = self.verbose))
        callbacks.append(KeepBestModelCallback(monitor = monitor, mode = "max", verbose = self.verbose))

        if self.verbose > 1:
            callbacks.append(BestStringCallback(self, generator))
            
        if self.verbose > 2:
            self.model.stateful_metric_names.append("loss")
            for key in self.losses:
                self.model.stateful_metric_names.append(key + "_loss")
            fit_verbose = 1
        else:
            fit_verbose = 0
        
        self.model.fit_generator(generator, steps_per_epoch = self.steps, epochs = 1000, callbacks = callbacks, verbose = fit_verbose)
        self._store_weights()


    def score_ngram(self, ngram):
        
        print(ngram)

        ngram = np.array([[self.dictionary[w] for w in ngram.split(" ")]])
        output = self.orig_model.predict(ngram)[0]
        if output.ndim == 2:
            output = output[-1]
        mean = np.mean([output[i] for i in self.indices])

        print(mean)
        return mean

    def get_best_ngram(self):
        return " ".join(self.rev_dictionary[w] for w in self._get_best_ngram())


class _OptimizerProbabilities(_OptimizerParametrical):
    steps = 200
    def __init__(self, **kwargs):
        super(_OptimizerProbabilities, self).__init__(**kwargs)

    def _prep(self):
        super(_OptimizerProbabilities, self)._prep()
        
        self.argmax = Argmax(name = "argmax")
        
        tmp = Dense(units = self.configs["emb"]["output_dim"], activation = "linear", use_bias = False)
        self.embedding = TimeDistributed(tmp, name = "emb_exp", trainable = False, weights = self.weights["emb"])

        self.logits = Embedding(input_dim = self.length, output_dim = len(self.dictionary), name = "logits")
    
    def _store_weights(self):
        with open(self.weightpath, "wb") as handle:
            _pickle.dump(self.logits.get_weights()[0], handle)

    def _get_output(self, probabilities):
        encoded = self.encoder(self.embedding(probabilities))
        if self.with_projection:
            encoded = self.projection(encoded)
        return self.selector(encoded)
            
    def _build(self):
    
        logits = self.logits(self.dummy_input)

        selection_soft = self._get_output(self._get_probabilities(logits))
        
        self.outputs.append(NameLayer("s")(selection_soft))
        self.outputs.append(NameLayer("L")(selection_soft))
        self.outputs.append(NameLayer("o_logits")(logits))
        
        if self.calc_hard:
            argmax = self.argmax(logits)
            self.outputs.append(NameLayer("o_argmax")(argmax))

            selection_hard = self._get_output(argmax)
            self.outputs.append(NameLayer("h")(selection_hard))
    
    def _get_best_ngram(self):
        return self.logits.get_weights()[0].argmax(axis = -1)
    
class _OptimizerEmbedding(_OptimizerParametrical):
    patience = 25
    steps = 20

    def __init__(self, **kwargs):
        super(_OptimizerEmbedding, self).__init__(**kwargs)
    
    def _store_weights(self):
        with open(self.weightpath, "wb") as handle:
            _pickle.dump(self.embedding.get_weights()[0], handle)
    
    def _prep(self):
        super(_OptimizerEmbedding, self)._prep()
        self.orig_emb = K.constant(self.weights["emb"][0])

        self.embedding = Embedding(\
            input_dim = self.length, 
            output_dim = self.configs["emb"]["output_dim"],
            name = "emb_exp")

        if self.calc_hard:
            self.snap = SnapToClosestLayer(self.orig_emb, mode = "max", name = "snap")
            self.cosine = PairwiseCosinesLayer(self.orig_emb, name = "cosine")
            self.max = Lambda(lambda x:K.max(x, axis = -1), output_shape = lambda shape:shape[:-1], name = "max")

    def _get_output(self, embedded):
        encoded = self.encoder(embedded)
        
        if self.with_projection:
            encoded = self.projection(encoded)
        return self.selector(encoded)
    
    def _get_best_ngram(self):
        similarities = cosine_similarity(self.embedding.get_weights()[0], self.weights["emb"][0])
        return similarities.argmax(-1)

class OptimizerEmbedding(_OptimizerEmbedding):
    name = "embfree"

    def _build(self):
        
        embedded = self.embedding(self.dummy_input)
        selection_soft = self._get_output(embedded)

        self.outputs.append(NameLayer("s")(selection_soft))
        self.outputs.append(NameLayer("L")(selection_soft))
        self.outputs.append(NameLayer("o_emb")(embedded))

        if self.calc_hard:
            cosines = self.cosine(embedded) # (length, vocab)
            maxcosines = self.max(cosines)
            
            self.outputs.append(NameLayer("c")(maxcosines))
            
            selection_hard = self._get_output(self.snap(cosines))
            self.outputs.append(NameLayer("h")(selection_hard))
        

        
class OptimizerGumbel(_OptimizerProbabilities):
    name = "gumbel"
    patience = 200

    def __init__(self, max_temperature = 2.0, min_temperature = 0.1, annealing_factor = 0.99995, batchsize = 64, **kwargs):
        super(OptimizerGumbel, self).__init__(batchsize = batchsize, **kwargs)
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.annealing_factor = annealing_factor
    
    def _prep(self):
        super(OptimizerGumbel, self)._prep()
        self.softmax = Activation("softmax", name = "softmax")
        self.gumbel = GumbelSoftmaxLayer(name = "gumbel")
        self.temperature = Input((1,), name = "i_temp")

    def _get_probabilities(self, logits):
        softmax = self.softmax(logits)
        gumbel = self.gumbel([softmax, self.temperature])
        
        self.outputs.append(NameLayer("o_gumbel")(gumbel))
        self.outputs.append(NameLayer("o_softmax")(softmax))
        self.outputs.append(NameLayer("t")(self.temperature))

        self.inputs.append(self.temperature)
        return gumbel

    def _get_generator(self):
        base_generator = self._base_generator()
        temperature_generator = self._temperature_generator()
        
        while True:
            dummy_in, dummy_out = next(base_generator)
            temperature = next(temperature_generator)
            dummy_in["i_temp"] = np.ones((self.batchsize, 1)) * temperature

            yield(dummy_in, dummy_out)


class OptimizerSoftmax(_OptimizerProbabilities):
    name = "softmax"
    patience = 100
    
    def _prep(self):
        super(OptimizerSoftmax, self)._prep()
        self.softmax = Activation("softmax", name = "softmax")

    def _get_probabilities(self, logits):
        softmax = self.softmax(logits)
        self.outputs.append(NameLayer(name = "o_softmax")(softmax))
        return softmax

class OptimizerLogits(_OptimizerProbabilities):
    name = "logits"
    patience = 100
    def _get_probabilities(self, logits):
        return logits


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest = "optimizer", type = str, 
        choices = ("search", "embfree", "logits", "gumbel", "softmax",))
    parser.add_argument(dest = "modality", type = str, choices = ("T", "I"))

    parser.add_argument("--target_layer", type = str, choices = ("output", "hidden"), default = "output")
    parser.add_argument("--model", type = str, default = os.path.join(paths.MODELDIR, "IMAGINET_arch.hdf5"), dest = "model")
    parser.add_argument("--dictionary", type = str, default = paths.DICTIONARYPATH, dest = "dictionary")
    parser.add_argument("--textdir", type = str, default = paths.TEXTDIR, dest = "textdir")
    parser.add_argument("--csvdir", type = str, default = paths.CSVDIR, dest = "csvdir")
    parser.add_argument("--weightdir", type = str, default = paths.WEIGHTDIR, dest = "weightdir")
    parser.add_argument("--length", type = int, default = 5, dest = "length")
    parser.add_argument("--verbose", type = int, default = 2, dest = "verbose")
    args = parser.parse_args()
    return vars(args)

def get_optimizer_class(optimizer):
    if optimizer == "search":
        return OptimizerSearch
    if optimizer == "embfree":
        return OptimizerEmbedding
    if optimizer == "gumbel":
        return OptimizerGumbel
    if optimizer == "softmax":
        return OptimizerSoftmax
    if optimizer == "logits":
        return OptimizerLogits
    else:
        raise Exception("Unknown optimizer:", optimizer)

def get_indices(num_draws = 80, max_index = 1024, min_num = 1, max_num = 1, not_these = []):
    state = np.random.RandomState(123)
    not_these = set(not_these)

    options = list(range(max_index))
    indices = set()

    while len(indices) < num_draws:
        num_neurons = state.randint(low = min_num, high = max_num + 1)
        tmp = None
        while tmp is None or tmp in not_these or tmp in indices:
            choice = state.choice(options, size = (num_neurons,), replace = False)
            tmp = tuple(sorted(choice))
        indices.add(tmp)

    indices = tuple(sorted(indices, key = lambda x:x[0]))
    return indices

def prettyprint(indices, ngram, targetword, score):
    print("OPT", end = "\t")
    print(" ".join(str(i) for i in indices), end = "\t")
    print(ngram, end = "\t")
    print(targetword, end = "\t")
    print(score, flush = True)

def get_index_groups(args, num_draws = 81):
    state = np.random.RandomState(123)
    bboptimizer = OptimizerSearch(indices = [[0]], **args)
    bboptimizer.build()

    subset = state.choice(list(range(bboptimizer.ngrams.shape[0])), size = (100000,))
    subset = bboptimizer.ngrams[subset]
    activations = bboptimizer.model.predict(subset, verbose = args["verbose"] > 2).T

    links = linkage(activations, method='complete', metric='correlation')
    clusters = [(i,) for i in range(activations.shape[0])]
    current_clusters = set([(i,) for i in range(activations.shape[0])])

    for link in links[:-(num_draws-1)]:
        left = clusters[int(link[0])]
        right = clusters[int(link[1])]
        new = sorted(list(left + right))
        new = tuple(new)

        clusters.append(new)
        current_clusters.add(new)
        current_clusters.remove(left)
        current_clusters.remove(right)

    assert len(set(sum(current_clusters, tuple([])))) == activations.shape[0]

    sizes = [len(cluster) for cluster in current_clusters]
    if args["verbose"] > 0:
        print("Max/min/median/mean cluster size: {}/{}/{}/{}".format(\
            max(sizes), min(sizes), int(np.median(sizes)), round(np.mean(sizes), 1)))
    return sorted(list(current_clusters))

if __name__ == "__main__":

    args = parse_arguments()

    optimizer = args.pop("optimizer")
    optimizer_class = get_optimizer_class(optimizer)

    if args["target_layer"] == "output":
        INDICES = get_indices()
        INDICES += get_indices(not_these = INDICES)

    else:
        INDICES = get_index_groups(args)

    K.clear_session()
    targetword = "_"

    if optimizer == "search":
        optimizer = optimizer_class(indices = INDICES, **args)
        optimizer.build()
        optimizer.train()

        
        ngrams = optimizer.get_best_ngrams()
        scores = optimizer.score_best_ngrams()

        for indices, ngram, score in zip(INDICES, ngrams, scores):
            if args["modality"] == "T" and len(indices) == 1 and args["target_layer"] == "output":
                targetword = optimizer.rev_dictionary[indices[0]]

            prettyprint(indices, ngram, targetword, score)

    else:
        for indices in INDICES:
            optimizer = optimizer_class(indices = indices, **args)
            optimizer.build()
            optimizer.train()

            if args["modality"] == "T" and len(indices) == 1 and args["target_layer"] == "output":
                targetword = optimizer.rev_dictionary[indices[0]]
            
            ngram = optimizer.get_best_ngram()
            prettyprint(indices, ngram, targetword, optimizer.score_ngram(ngram))
            K.clear_session()

    K.clear_session()    
    




