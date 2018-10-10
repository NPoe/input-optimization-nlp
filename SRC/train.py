import numpy as np
np.random.seed(123)

import os
import sys
import _pickle

import argparse

from keras.layers import *
from keras.models import *
from keras.callbacks import *

from sklearn.metrics.pairwise import cosine_similarity

import paths

class Trainer:

    def __init__(self, verbose, modeldir, dictionary, textdir, imgdir, alpha, embeddingdim, hiddendim, batchsize):
        
        self.modeldir = modeldir
        self.dictionary = dictionary
        self.batchsize = batchsize
        self.imgdir = imgdir
        self.textdir = textdir
        self.verbose = verbose

        self.embeddingdim = embeddingdim
        self.hiddendim = hiddendim
        self.batchsize = batchsize

        self.losses = {}
        self.metrics = {}
        self.loss_weights = None    
        
        self.losses["I"] = "cosine_proximity"
        self.losses["T"] = "sparse_categorical_crossentropy"
        self.metrics["T"] = "accuracy"
        self.loss_weights = {"T": alpha, "I": 1-alpha}
        
        self.prepped = False
        self.built = False
    
        self.model_name = "IMAGINET_train"
 
    def test_generator(self):
        generator = self.generator("train")
        x, y = next(generator)

        print("Generator test")
        print("Input:", " ".join(self.rev_dictionary[word] for word in x[0]))
        print("T target:", " ".join(self.rev_dictionary[word] for word in y["T"][0].squeeze()))
        print("I target is an image representation with shape:", y["I"][0].shape)
        print()
    
    def img_accuracy(self, epoch, logs):
        
        txts = self.txts["val"]
        imgs = self.imgs["val"]
        
        maxlen = max([len(txt) for txt in txts])
        txts = np.array([[0] * (maxlen - len(txt)) + txt for txt in txts])
        predictions = self.modality_models["I"].predict(txts, verbose = 0, batch_size = self.batchsize)

        similarities = cosine_similarity(predictions, imgs)
        
        K = 10
        topK = np.argpartition(similarities, kth = -K, axis = 1)[:,-K:]
        
        matches = 0
        for j in range(K):
            print("Acc @", j + 1, end = " ", flush = True)
            for i in range(predictions.shape[0]):
                if topK[i,j] == i//5:
                    matches += 1

            print(matches, "/", predictions.shape[0], "=", round(matches / predictions.shape[0], 3))
            
    def prep(self):
        if isinstance(self.dictionary, str):
            self.dictionary = _pickle.load(open(self.dictionary, "rb"))

        self.rev_dictionary = {self.dictionary[key]: key for key in self.dictionary}
        
        self.txts = {}
        self.imgs = {}
        
        for dataset in ("train", "val", "test"):
            self.txts[dataset] = []    
            with open(os.path.join(self.textdir, dataset + ".txt")) as handle:
                for line in handle:
                    tmp = line.strip().split()
                    self.txts[dataset].append(\
                        [self.dictionary.get(word, self.dictionary["<UNK>"]) for word in tmp])
            
            with open(os.path.join(self.imgdir, dataset + ".npy"), "rb") as handle:
                self.imgs[dataset] = np.load(handle)
                
            assert len(self.txts[dataset]) == self.imgs[dataset].shape[0] * 5

        self.prepped = True

    def spe(self, dataset):
        tmp = len(self.txts[dataset])
        return tmp // self.batchsize + int(tmp%self.batchsize > 0)

    def generator(self, dataset, shuffle = True):
    
        indices = list(range(len(self.txts[dataset])))

        while True:
            if shuffle:
                np.random.shuffle(indices)

            for i in range(0, len(indices), self.batchsize):
                batch_idx = indices[i:min(i+self.batchsize, len(indices))]
                batch_sents = [self.txts[dataset][idx] for idx in batch_idx]

                maxlen = max(len(sent) for sent in batch_sents)
                batch_sents = np.array([[0] * (maxlen-len(sent)) + sent for sent in batch_sents])
                
                labels = {}
                end_tags = np.ones_like(batch_sents[:,:1]) * self.dictionary["<END>"]
                batch_lm_target = np.concatenate([batch_sents[:,1:], end_tags], axis = 1)
                labels["T"] = np.expand_dims(batch_lm_target, -1)

                batch_imgs = np.stack([self.imgs[dataset][idx//5] for idx in batch_idx], axis = 0)
                labels["I"] = batch_imgs

                yield(batch_sents, labels)
    
    
    def build(self):
        if not self.prepped:
            self.prep()
        
        print("Building main model")

        self.inputs = Input((None,), name = "input")

        self.embedding = Embedding(\
            input_dim = len(self.dictionary),
            output_dim = self.embeddingdim,
            mask_zero = True,
            name = "emb")
        
        self.encoded = {}
        self.encoders = {}

        for modality in ["I", "T"]:
            self.encoders[modality] = GRU(units = self.hiddendim, return_sequences = modality == "T", name = "enc_" + modality)
        
        self.encoded = {modality: self.encoders[modality](self.embedding(self.inputs)) for modality in self.encoders.keys()}

        self.projections = {}
        self.projections["T"] = TimeDistributed(Dense(units = len(self.dictionary), activation = "softmax"), name = "T")
        self.projections["I"] = Dense(units = self.imgs["train"].shape[-1], activation = "linear", name = "I")
        
        self.outputs = {modality: self.projections[modality](self.encoded[modality]) for modality in self.projections.keys()}
        
        self.modality_models = {modality: Model([self.inputs], [self.outputs[modality]]) for modality in self.outputs.keys()}
        self.model = Model([self.inputs], [self.outputs[modality] for modality in self.outputs.keys()])
        
        self.model.compile(\
            loss = self.losses, 
            optimizer = "adam", 
            metrics = self.metrics, 
            loss_weights = self.loss_weights)
        
        self.model.summary()
        self.built = True
        

    def load(self):
        if not self.built:
            self.build()

        self.model.load_weights(os.path.join(self.modeldir, self.model_name + ".hdf5"))
        if self.verbose:
            self.model.summary()
            self.evaluate()

    def evaluate(self):
        print("Evaluating")
        results = self.model.evaluate_generator(self.generator("val"), steps = self.spe("val"), verbose = 0)
        for metric, result in zip(self.model.metrics_names, results):
            print(metric, ":", round(result, 5))


    def train(self):
        
        if not self.built:
            self.build()

        if self.verbose:
            self.test_generator()

        callbacks = []
        
        callbacks.append(LambdaCallback(on_epoch_end = lambda epoch, logs: self.img_accuracy(epoch, logs)))
        callbacks.append(EarlyStopping(monitor = "val_loss", patience = 10))
        callbacks.append(CSVLogger(os.path.join(self.modeldir, self.model_name + ".csv")))
        callbacks.append(ModelCheckpoint(\
            os.path.join(self.modeldir, self.model_name + ".hdf5"), 
            save_best_only = True,
            monitor = "val_loss"))

        self.model.fit_generator(\
            self.generator("train"), 
            steps_per_epoch = self.spe("train"), 
            epochs = 1000, 
            verbose = self.verbose,
            validation_data = self.generator("val"), 
            validation_steps = self.spe("val"), 
            callbacks = callbacks)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", type = str, default = paths.MODELDIR, dest = "modeldir")
    parser.add_argument("--dictionary", type = str, default = paths.DICTIONARYPATH, dest = "dictionary")
    parser.add_argument("--textdir", type = str, default = paths.TEXTDIR, dest = "textdir")
    parser.add_argument("--imgdir", type = str, default = paths.IMGDIR, dest = "imgdir")
    parser.add_argument("--alpha", type = float, default = 0.1, dest = "alpha")
    parser.add_argument("--batchsize", type = int, default = 16, dest = "batchsize")
    parser.add_argument("--hiddendim", type = int, default = 1024, dest = "hiddendim")
    parser.add_argument("--embeddingdim", type = int, default = 1024, dest = "embeddingdim")
    parser.add_argument("--verbose", type = int, default = 2, dest = "verbose")

    args = parser.parse_args()
    return vars(args)
    
if __name__ == "__main__":
    args = parse_arguments()

    trainer = Trainer(**args)
    trainer.train()

