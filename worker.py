import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
import os
import copy

def to_img(x):
    x = x.clamp(0, 1)
    return x

class AEDetector:
    def __init__(self, model, device, path, p=1):
        """
        Error based detector.
        Marks examples for filtering decisions.
        
        model : pytorch model class
        device : torch.device
        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.model = model
        self.model.load_state_dict(torch.load(path))
        self.path = path
        self.p = p
        
        self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def mark(self, X):
       
        if torch.is_tensor(X):
            X_torch = X
        else :
            X_torch = torch.from_numpy(X)
            
        diff = torch.abs(X_torch - 
                         self.model(X_torch.to(self.device)).detach().cpu())
        marks = torch.mean(torch.pow(diff, self.p), dim = (1,2,3))
        
        return marks

    def print(self):
        return "AEDetector:" + self.path.split("/")[-1]
    
class IdReformer:
    def __init__(self, path="IdentityFunction"):
        """
        Identity reformer.
        Reforms an example to itself.
        """
        self.path = path
        self.heal = lambda X: X

    def print(self):
        return "IdReformer:" + self.path
    
class SimpleReformer:
    def __init__(self, model, device, path):
        """
        Reformer.
        Reforms examples with autoencoder. Action of reforming is called heal.

        path: Path to the autoencoder used.
        """
        self.model = model
        self.model.load_state_dict(torch.load(path))
        #self.model = load_model(path)
        self.path = path
        self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
 
    def heal(self, X):
        #X = self.model.predict(X)
        #return np.clip(X, 0.0, 1.0)
 
        if torch.is_tensor(X):
            X_torch = X
        else :
            X_torch = torch.from_numpy(X)
        
        X = self.model(X_torch.to(self.device)).detach().cpu()
        
        return torch.clamp(X, 0.0, 1.0)

    def print(self):
        return "SimpleReformer:" + self.path.split("/")[-1]

class Classifier:
    def __init__(self, model, device, classifier_path, device_ids = [0]):
        """
        Keras classifier wrapper.
        Note that the wrapped classifier should spit logits as output.

        model : pytorch model class
        device : torch.device
        classifier_path: Path to Keras classifier file.
        """
        self.path = classifier_path
        
        self.model = model
        self.model.load_state_dict(torch.load(classifier_path))
        
        self.softmax = nn.Softmax(dim = 1)
 
        self.device = device
    
        if len(device_ids) > 1 :
            self.model = nn.DataParallel(self.model, device_ids = device_ids)
    
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def classify(self, X, option="logit", T=1):
        
        if torch.is_tensor(X):
            X_torch = X
        else :
            X_torch = torch.from_numpy(X)
          
        X_torch = X_torch.to(self.device)
        
        if option == "logit":
            
            return self.model(X_torch).detach().cpu()
        if option == "prob":
            logits = self.model(X_torch) / T
            logits =  self.softmax(logits)
            return logits.detach().cpu()
            
    def print(self):
        return "Classifier:"+self.path.split("/")[-1]
    
    
class AttackData:
    def __init__(self, examples, labels, name=""):
        """
        Input data wrapper. May be normal or adversarial.

        examples: object of input examples.
        labels: Ground truth labels.
        """

        self.data = examples
        self.labels = labels
        self.name = name

    def print(self):
        return "Attack:"+self.name

def operate(reformer, classifier, inputs, filtered = True):
    
    X = inputs
 
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    
    if filtered :
        X_prime = reformer.heal(X)
        Y_prime = classifier.classify(X_prime)
        
    else :
        Y_prime = classifier.classify(X)

    return Y_prime  

def filters(detector, data, thrs):
    """
    untrusted_obj: Untrusted input to test against.
    thrs: Thresholds.

    return:
    all_pass: Index of examples that passed all detectors.
    collector: Number of examples that escaped each detector.
    """
    collector = []
    all_pass = np.array(range(10000))
    
    marks = detector.mark(data)

    np_marks = marks.numpy()
    np_thrs = thrs.numpy()

    idx_pass = np.argwhere(np_marks < np_thrs)
    collector.append(len(idx_pass))
    all_pass = np.intersect1d(all_pass, idx_pass)

    all_pass = torch.from_numpy(all_pass)

    return all_pass, collector