    if p == np.inf:
        """
        x = np.asarray(x)
        if x.shape[-1] != self.m:
        """
output = asarray(DataReader('GDPC1', 'fred',
                            start=start1, end=end1)).squeeze()
investment = asarray(DataReader('GPDIC96', 'fred',
                                start=start1, end=end1)).squeeze()
consumption = asarray(DataReader('PCECC96', 'fred',
        self.A = asarray([[1.0, 0.0, 1.0], [0.0, 1.0, -4.0]])
        self.b = asarray([-1.0, 0.0])
        self.bounds = ((None, None), (0.1, 2.9), (0.7, None))
from numpy.testing import *
from numpy import (arange, asarray, empty, float64, zeros)
    y = np.asarray(y, dtype=np.int32)
    # the OpenCV wrapper happy:
    # If a out_dir is given, set it:

    X = _convert_to_double(np.asarray(X, order='c'))
    out['asps'] = np.asarray((asps, asps_err))
    return out, extra_out
    out = {}
            extra_img['refs'] = refs
    out['pips'] = np.asarray((pips, pips_err))
        yj = asarray(yj)

                          [0.0, 0.0, 1.0, -2.0, -2.0]])
        self.b = asarray([5.0, -3.0])

    out['vess'] = np.asarray((ves, PIX_ERR*np.ones_like(ves)))
    out['asps'] = np.asarray((asp, PIX_ERR*np.ones_like(asp)))
    out = {}
    out['piprads'] = np.asarray(
                            (piprad, PIX_ERR*np.sqrt(2)*np.ones_like(piprad)))
        self.b = asarray([5.0, -3.0])

        self.pt = (-1.725, 2.9, 0.725)
        self.A = asarray([[1.0, 0.0, 1.0], [0.0, 1.0, -4.0]])
        self.b = asarray([-1.0, 0.0])
    >>> p3 = npy.asarray([1,1])
    >>> print area_of_triangle(p1,p2,p3)
    """
    >>> xy = npy.asarray([[0,0],[0,1],[1,1],[1,0]])
    >>> print avg_speed(xy)
    out['asps'] = np.asarray((asp, PIX_ERR*np.ones_like(asp)))
    out = {}
    out['piprads'] = np.asarray(
                            (piprad, PIX_ERR*np.sqrt(2)*np.ones_like(piprad)))
    if img.dtype == np.int32:
        self.pt = (-1.725, 2.9, 0.725)
        self.A = asarray([[1.0, 0.0, 1.0], [0.0, 1.0, -4.0]])
        self.b = asarray([-1.0, 0.0])
        self.bounds = ((None, None), (0.1, 2.9), (0.7, None))
from numpy.testing import *
        
    def test_asarray(self):
        c = ConfusionMatrix([[1,2],[3,4]])
        self.assertTrue(np.all(c.asarray() == np.array([[1,2],[3,4]])))
        
            else:
                VI = np.asarray(VI, order='c')
            [VI] = _copy_arrays_if_base_present([VI])
wages = asarray(DataReader('WASCUR', 'fred',
                                start=start1, end=end1)).squeeze()
       Optional mirroring with "mirrored" argument.'''
    points,newRefLine = np.asarray(points), np.asarray(newRefLine)
    dx,dy = (newRefLine[1]-newRefLine[0])
    '''Subtract off the mean from x'''
    x = np.asarray(x)
        self.bounds = ((None, None), (0.1, 2.9), (0.7, None))
from numpy.testing import *
from numpy import (arange, asarray, empty, float64, zeros)

        self.pt = (-0.09, 0.03, 0.25, -0.19, 0.03)
    def test_asarray(self):
        c = ConfusionMatrix([[1,2],[3,4]])
        self.assertTrue(np.all(c.asarray() == np.array([[1,2],[3,4]])))
        
        self.assertTrue(np.all(cb.asarray() == np.array([[0.25,0.75],[0.5,0.5]])))
