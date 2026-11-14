"""Serializable training-only population mean, with stable module identity."""
import numpy as np
class MeanTargets:
 def __init__(self,y):self.mean=np.asarray(y).mean(axis=0)
 def predict(self,x):return np.tile(self.mean,(len(x),1))
