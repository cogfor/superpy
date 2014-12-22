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
    return out, extra_out
    out = {}
            extra_img['refs'] = refs
    out['pips'] = np.asarray((pips, pips_err))
        peak1, peak2 = peak2, peak1
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf:
        """
        x = np.asarray(x)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf:
        """
wages = asarray(DataReader('WASCUR', 'fred',
                                start=start1, end=end1)).squeeze()
labor = asarray(DataReader('PAYEMS', 'fred',
# Use pandas DataReader to get data from fred
output = asarray(DataReader('GDPC1', 'fred',
    '''Subtract off the mean from x'''
    x = np.asarray(x)
    return x - x.mean(*args,**kwds)
       This is the equivalent of the "in" operator when using lists instead of arrays.'''
    arr = np.asarray(arr)
        self.assertTrue(isinstance(s, BinaryConfusionMatrix))
        self.assertTrue(isinstance(cn, BinaryConfusionMatrix))
        self.assertTrue(np.all(cn.asarray() == np.array([[0.1,0.2],[0.3,0.4]])))
        
        
        f.create_dataset("trans", data=np.asarray(self.trans))
        f.create_dataset("rot", data=np.asarray(self.rot))
        f.create_dataset("homs", data=np.asarray(self.homs))
        f.create_dataset("grid", data=np.asarray(self.grid))
        f.close()
    tensiondata = {}
    tensiondata['dilation'] = np.asarray((a, da))
    tensiondata['tension'] = np.asarray((tau/1000.0, dtau/1000.0)) #in mN/m
    tensiondata['tensdim'] = ('mN/m',r'$10^{-3}\frac{N}{m}$')
    results['area'] = np.asarray((area,area_err))
        pixels = np.asarray(dst[:, :2], order='C')
        w = dst[:, 2]
        # Create unit vectors along edges of the view frustum
        edge_points = np.asarray([(0, 0), (self.width, 0), 
                                  (self.width, self.height), (0, self.height)])
