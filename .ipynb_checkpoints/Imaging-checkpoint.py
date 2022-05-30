import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from typing import Tuple, Union
from pathlib import Path

def plot(X: np.array, figsize: Tuple[int] = (12, 7), cmap: str = 'Greys',
         outputPath: Union[str, Path] = None):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize)
    plt.axis('off')
    im = ax.imshow(X[0], cmap = cmap, origin = 'lower')
    plt.show()
    
    if outputPath:
        outputPath = Path(outputPath) if type(outputPath) == str else outputPath
        fig.savefig(path, bbox_inches = 'tight')
        
    return fig, ax

class Imaging:
    def __init__(self):
        self.methods = {
            'gasf': GASF(),
            'gadf': GADF(),
            'mtf' : MTF(),
            'rp'  : RP()
        }
            
    def apply(self, method: str, X: np.array, **kwargs):
        pic_matrix = self.methods[method].apply(X, **kwargs)
        return pic_matrix

class ImagingMethod:
    def __init__(self):
        pass
    
    def apply(self, X: np.array, **kwargs):
        return self._method(X, **kwargs)
    
    def __method(self, **kwargs):
        raise NotImplementedError
        
class GADF(ImagingMethod):
    def _method(self, X, **kwargs):
        gadf = GramianAngularField(method = 'difference', **kwargs)
        
        return gadf.fit_transform(X)
    
class GASF(ImagingMethod):
    def _method(self, X: np.array, **kwargs):
        gasf = GramianAngularField(method = 'summation', **kwargs)
        
        return gasf.fit_transform(X)
    
class MTF(ImagingMethod):
    def _method(self, X: np.array, **kwargs):
        mtk = MarkovTransitionField(**kwargs)
        
        return mtk.fit_transform(X)
    
class RP(ImagingMethod):
    def _method(self, X: np.array, **kwargs):
        rp = RecurrencePlot(**kwargs)
        
        return rp.fit_transform(X)
    
            