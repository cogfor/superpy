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
       x can also be a list of vectors'''
    x = np.asarray(x)
    sh = list(x.shape)
       Optional mirroring with "mirrored" argument.'''
    points,newRefLine = np.asarray(points), np.asarray(newRefLine)
    minX, maxX = np.min(X), np.max(X)
    # Learn the model. Remember our function returns Python lists,
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
            b = b.asarray()
        a = self.asarray()
        #if isinstance(b, MetaArray):
            #b = b.asarray()
        #return self.asarray() != b
        P = np.asarray(self.P, dtype=np.float)
        dst = blas.dgemv(np.array([P], order='C'), src)
        pixels = np.asarray(dst[:, :2], order='C')
        w = dst[:, 2]
        # Create unit vectors along edges of the view frustum
                                start=start1, end=end1)).squeeze()
import pandas as pd
from numpy import asarray
from pandas.io.data import DataReader
                                start=start1, end=end1)).squeeze()
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p==np.inf:
    """Compute the L**p distance between x and y"""
        peak1, peak2 = peak2, peak1
    return np.asarray((peak1, peak2))
    
    out['metrics'] = np.asarray((metrics, metrics_err))
    '''Compute the sqrt of the mean of the squares (RMS)'''
                #print "matches found"
        self.data = np.asarray(data)
        self.n, self.m = np.shape(self.data)

    x = np.asarray(x)
                self._data = data.asarray()
            elif isinstance(data, tuple):  ## create empty array with specified shape
        ma = MetaArray._h5py_metaarray.MetaArray(file=fileName)
        self._data = ma.asarray()._getValue()
        self._info = ma._info._getValue()
                kwargs[key] = np.asarray((averval, avererr))
    return kwargs
    tensiondata = {}
        dref = np.asarray([PIX_ERR, 0])
        if (fit[-1] != 1) or (err == None) or (sum(err[2:]) >= mismatch):
                            start=start1, end=end1)).squeeze()
investment = asarray(DataReader('GPDIC96', 'fred',
                                start=start1, end=end1)).squeeze()
consumption = asarray(DataReader('PCECC96', 'fred',
                                start=start1, end=end1)).squeeze()
    out['piprads'] = np.asarray((piprads, piprads_err))
    out['vess'] = np.asarray((vess, vess_err))
    out['asps'] = np.asarray((asps, asps_err))
    return out, extra_out
    out = {}
        cart_coords = misctools.spherical_to_cartesian(
            np.asarray([radius, theta, phi]))
        cart_coords = cart_coords[[0, 2]].T
                desc_l = np.asarray(im_left.descriptors)[idx_l]
                kp_r = np.asarray(im_right.keypoints)[idx_r]
    return x - x.mean(*args,**kwds)
       This is the equivalent of the "in" operator when using lists instead of arrays.'''
    arr = np.asarray(arr)
    subarr = np.asarray(subarr)
    if subarr.shape!=arr.shape[1:]:
        """
    x = np.asarray(x)
    y = np.asarray(y)
    """
    x = np.asarray(x)
    >>> print mrdo_speed(xy)
        measures.append((pl,avg,mrdo,hull_surf,hull_d,hull_density,hull_drel))
    measures = npy.asarray(measures)

    >>> p2 = npy.asarray([1,0])
            a, b, x0, s = fit[0]
            refs = np.append(refs, np.asarray((x0 - s, refx)), axis = 1)
            refs_err = np.append(refs_err, sum(err[2:]))
                    k*k*dk*dk*(k*x0+b-y0)**2/(1+k*k))/(1+k*k)
    return np.asarray((np.fabs(dist), dist_err))
        P = np.asarray(self.P, dtype=np.float)
        dst = blas.dgemv(np.array([P], order='C'), src)
        pixels = np.asarray(dst[:, :2], order='C')
        w = dst[:, 2]
        # Create unit vectors along edges of the view frustum
    x = np.asarray(x)
    sh = list(x.shape)
       Optional mirroring with "mirrored" argument.'''
    points,newRefLine = np.asarray(points), np.asarray(newRefLine)
    dx,dy = (newRefLine[1]-newRefLine[0])
                          [0.0, 0.0, 1.0, 1.0, -2.0],
        self.pt = (1.0, 1.0, 1.0, 1.0, 1.0)
        self.A = asarray([[1.0, 1.0, 1.0,  1.0,  1.0],
    results['metrics'] = argsdict['metrics']
    tensiondata['dilation'] = np.asarray((alpha,alpha_err))
    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')
    """
    Y = np.asarray(Y, order='c')
    is_valid_y(Y, throw=True, name='Y')
        rx = np.tile(refx,N)
            a, b, x0, s = fit[0]
            refs = np.append(refs, np.asarray((x0 + s, refx)), axis = 1)
            #taking err_x0+err_s while ref = x0+s
        if (fit[-1] != 1) or (err == None) or (sum(err[2:]) >= mismatch):
import pandas as pd
from numpy import asarray
from pandas.io.data import DataReader
                                start=start1, end=end1)).squeeze()
        self.tri_soma[:,:] = self.tri_soma[:,:]*R
  rCam.trans=np.asarray(tvecs)

      #convert cvMat to numpy array
      tmpIm = np.asarray(cvFinal)
      
        if x.shape[-1] != self.m:
        """
        x = np.asarray(x)
        if np.shape(x)[-1] != self.m:
        elif len(np.shape(r))==1:
    y = np.asarray(y, dtype=np.int32)
    # the OpenCV wrapper happy:
    # If a out_dir is given, set it:

    X = _convert_to_double(np.asarray(X, order='c'))
    from scipy.interpolate import interp1d
    dcb_radiance = asarray(dcb_radiance)
    assert (dcb_radiance.ndim == 3), 'Input "dcb_radiance" must have three dimensions.'

      #convert cvMat to numpy array
      tmpIm = np.asarray(cvFinal)
      
    Hom[2,2]=1
    arr = np.asarray(arr)
    subarr = np.asarray(subarr)
    if subarr.shape!=arr.shape[1:]:
        #return MetaArray(self.asarray() - b, info=self.infoCopy())

    x = np.asarray(x)
    y = np.asarray(y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.asarray(pressure), aver, mesg

    out['vess'] = np.asarray((ves, PIX_ERR*np.ones_like(ves)))
    out['asps'] = np.asarray((asp, PIX_ERR*np.ones_like(asp)))
    out = {}
                desc_l = np.asarray(im_left.descriptors)[idx_l]
                kp_r = np.asarray(im_right.keypoints)[idx_r]
                desc_r = np.asarray(im_right.descriptors)[idx_r]
                # Valid matches are those where y co-ordinate of p1 and p2 are
                #print "matches found"

        #model center
        self.center = npy.asarray((x0,y0),dtype=float)

            #create dataset
    sh = list(x.shape)
       Optional mirroring with "mirrored" argument.'''
    points,newRefLine = np.asarray(points), np.asarray(newRefLine)
    dx,dy = (newRefLine[1]-newRefLine[0])
    '''Subtract off the mean from x'''

    >>> p2 = npy.asarray([1,0])
        self.assertTrue(isinstance(cb, BinaryConfusionMatrix))
        self.assertTrue(isinstance(s, BinaryConfusionMatrix))
        self.assertTrue(isinstance(cn, BinaryConfusionMatrix))
                    k*k*dk*dk*(k*x0+b-y0)**2/(1+k*k))/(1+k*k)
    return np.asarray((np.fabs(dist), dist_err))

            extra_img['piprad'] = np.asarray((piprad, PIX_ERR))
            extra_img['profile'] = profile
        w = dst[:, 2]
        # Create unit vectors along edges of the view frustum
        edge_points = np.asarray([(0, 0), (self.width, 0), 
                                  (self.width, self.height), (0, self.height)])
                kp_l = np.asarray(im_left.keypoints)[idx_l]
from pandas.io.data import DataReader
                                start=start1, end=end1)).squeeze()
        self.tri_soma[:,:] = self.tri_soma[:,:]*R
        self.tri_soma[:,:] += npy.asarray([x,y,x,y,x,y])

    results['metrics'] = argsdict['metrics']
    tensiondata['dilation'] = np.asarray((alpha,alpha_err))
    tensiondata['tensdim'] = ('mN/m',r'$10^{-3}\frac{N}{m}$')
    tensiondata['tension'] = np.asarray((tau/1000.0, tau_err/1000.0)) #in mN/m
    tensiondata = {}
    measures = npy.asarray(measures)

    """
    >>> xy = npy.asarray([[0,0],[0,1],[1,1],[1,0]])
    >>> print mrdo_speed(xy)
            #taking err_x0+err_s while ref = x0+s
        if (fit[-1] != 1) or (err == None) or (sum(err[2:]) >= mismatch):
            refs = np.append(refs, np.asarray((centerest, refx)), axis = 1)
            refs_err = np.append(refs_err, 1)
        refy = np.asarray((np.argmax(filtered[:mid]), np.argmax(filtered[mid:])+mid))
    from scipy.interpolate import interp1d
    dcb_radiance = asarray(dcb_radiance)
    assert (dcb_radiance.ndim == 3), 'Input "dcb_radiance" must have three dimensions.'

      tmpIm = np.asarray(cvFinal)
      
    Hom[2,2]=1
    Homs.append(np.asarray(Hom,dtype=float))
    Rot.append(np.asarray(tmpRot,dtype=float))
from numpy.testing import *
from numpy import (arange, asarray, empty, float64, zeros)

        self.pt = (-0.09, 0.03, 0.25, -0.19, 0.03)
        self.A = asarray([[1.0, 3.0, 0.0, 0.0,  0.0],

    X = _convert_to_double(np.asarray(X, order='c'))


    XA = np.asarray(XA, order='c')

    out['vess'] = np.asarray((ves, PIX_ERR*np.ones_like(ves)))
    out['asps'] = np.asarray((asp, PIX_ERR*np.ones_like(asp)))
    out = {}
    out['piprads'] = np.asarray(
                kp_r = np.asarray(im_right.keypoints)[idx_r]
                desc_r = np.asarray(im_right.descriptors)[idx_r]
                # Valid matches are those where y co-ordinate of p1 and p2 are
                #print "matches found"
        self.data = np.asarray(data)
        f.close()
  rCam = Cam()
  rCam.cmat=np.asarray(cmat)
  rCam.dist=np.asarray(distCoeffs)
  rCam.rot=np.asarray(rvecs)
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf:
        """
        x = np.asarray(x)
                    y.append(c)
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

    tensiondata['tensdim'] = ('mN/m',r'$10^{-3}\frac{N}{m}$')
    results['area'] = np.asarray((area,area_err))
    results['volume'] = np.asarray((volume,volume_err))
    results['piprad'] = np.asarray((piprad,piprad_err))
                avererr = err.reshape((-1,aver)).mean(axis=1)
wages = asarray(DataReader('WASCUR', 'fred',
                                start=start1, end=end1)).squeeze()
labor = asarray(DataReader('PAYEMS', 'fred',
# Use pandas DataReader to get data from fred
output = asarray(DataReader('GDPC1', 'fred',
        f.create_dataset("trans", data=np.asarray(self.trans))
        f.close()
        f = h5.File(filename, 'w')
        f.create_dataset("trans", data=np.asarray(self.trans))
        f.create_dataset("rot", data=np.asarray(self.rot))
    """Compute the L**p distance between x and y"""
    x = np.asarray(x)
    y = np.asarray(y)
    if p==np.inf or p==1:
        """
    # Thanks to Leo Dirac for reporting:
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # Learn the model. Remember our function returns Python lists,
        self.assert_((mat.asArray() == 2*eye(10)).all())

        mat.setRow(9, [1, -1])
        self.assert_((mat.asArray() == self.laplacian).all())

        

        src[:, 3] = 1.0
        P = np.asarray(self.P, dtype=np.float)
        dst = blas.dgemv(np.array([P], order='C'), src)
    return np.sqrt(np.mean(np.asarray(x)**2,axis=axis))

       x can also be a list of vectors'''
    x = np.asarray(x)
    sh = list(x.shape)
    m, k = x.shape
    y = np.asarray(y)
    n, kk = y.shape
    """
    x = np.asarray(x)
        if isinstance(b, MetaArray):
            b = b.asarray()
        a = self.asarray()
        #if isinstance(b, MetaArray):
            #b = b.asarray()
        if (fit[-1] != 1) or (err == None) or (sum(err[2:]) >= mismatch):
            refs = np.append(refs, np.asarray((centerest, refx)), axis = 1)
            refs_err = np.append(refs_err, 1)
        rx = np.tile(refx,N)
            a, b, x0, s = fit[0]
        c = ConfusionMatrix([[1,2],[3,4]])
        self.assertTrue(np.all(c.asarray() == np.array([[1,2],[3,4]])))
        
        self.assertTrue(np.all(cb.asarray() == np.array([[0.25,0.75],[0.5,0.5]])))
        t = ConfusionMatrix([[4,6],[4,6]])
    out = {}
            extra_img['refs'] = refs
    out['pips'] = np.asarray((pips, pips_err))
        peak1, peak2 = peak2, peak1
    return np.asarray((peak1, peak2))
    >>> print area_of_triangle(p1,p2,p3)
    """
    >>> xy = npy.asarray([[0,0],[0,1],[1,1],[1,0]])
    >>> print avg_speed(xy)

        #return MetaArray(self.asarray() - b, info=self.infoCopy())

                self._info = data._info
                self._data = data.asarray()
            elif isinstance(data, tuple):  ## create empty array with specified shape
    results['volume'] = np.asarray((volume,volume_err))
    results['piprad'] = np.asarray((piprad,piprad_err))
                avererr = err.reshape((-1,aver)).mean(axis=1)
                kwargs[key] = np.asarray((averval, avererr))
    return kwargs
labor = asarray(DataReader('PAYEMS', 'fred',
# Use pandas DataReader to get data from fred
output = asarray(DataReader('GDPC1', 'fred',
                            start=start1, end=end1)).squeeze()
investment = asarray(DataReader('GPDIC96', 'fred',

            extra_img['piprad'] = np.asarray((piprad, PIX_ERR))
            extra_img['profile'] = profile
    out['piprads'] = np.asarray((piprads, piprads_err))
    out['vess'] = np.asarray((vess, vess_err))

            #create dataset
            data = npy.asarray(data)
            ds = hdf5_group.create_dataset(s['dataset_name'], data.shape, dtype=float)
        self.tri_halo[:,2:6] = self.tri_halo[:,2:6] * R[:,npy.newaxis]
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # Learn the model. Remember our function returns Python lists,
                        im = cv2.resize(im, sz)
        self.assertTrue(isinstance(s, BinaryConfusionMatrix))
        self.assertTrue(isinstance(cn, BinaryConfusionMatrix))
        self.assertTrue(np.all(cn.asarray() == np.array([[0.1,0.2],[0.3,0.4]])))
        
        
            else:
                V = np.asarray(V, order='c')
            dm = pdist(X, lambda u, v: seuclidean(u, v, V))
            else:
                VI = np.asarray(VI, order='c')

    xycoords = asarray(xycoords)
    assert (Nx == int(Nx)), 'Input "Nx" must be an integer type.'
        yj = asarray(yj)

        self.tri_soma[:,:] += npy.asarray([x,y,x,y,x,y])

        #model center
        self.center = npy.asarray((x0,y0),dtype=float)

    tensiondata['tension'] = np.asarray((tau/1000.0, tau_err/1000.0)) #in mN/m
    tensiondata = {}
    tensiondata['dilation'] = np.asarray((a, da))
    tensiondata['tension'] = np.asarray((tau/1000.0, dtau/1000.0)) #in mN/m
    tensiondata['tensdim'] = ('mN/m',r'$10^{-3}\frac{N}{m}$')
        elif len(np.shape(r))==1:
            r = np.asarray(r)
            n, = r.shape
    """
        self.assert_((mat.asArray() == eye(10)).all())
    X = _convert_to_double(np.asarray(X, order='c'))


    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')
    assert (dcb_radiance.ndim == 3), 'Input "dcb_radiance" must have three dimensions.'

    import matplotlib.gridspec as gridspec
    img = asarray(img)
        z = zip(ybad[i]+yj,xbad[i]+xj)
    Homs.append(np.asarray(Hom,dtype=float))
    Rot.append(np.asarray(tmpRot,dtype=float))
    Trans.append(np.asarray(tmpTrans,dtype=float))
    transDist=np.sqrt(tmpTrans[0,0]*tmpTrans[0,0]+tmpTrans[0,1]*tmpTrans[0,1])
    # so we use np.asarray to turn them into NumPy lists to make
        self.pt = (-0.09, 0.03, 0.25, -0.19, 0.03)
        self.A = asarray([[1.0, 3.0, 0.0, 0.0,  0.0],
                          [0.0, 0.0, 1.0, 1.0, -2.0],
        self.pt = (1.0, 1.0, 1.0, 1.0, 1.0)
        self.A = asarray([[1.0, 1.0, 1.0,  1.0,  1.0],
    if p == np.inf:
        """
        x = np.asarray(x)
        if x.shape[-1] != self.m:
        """
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

    y = np.asarray(y, dtype=np.int32)
    # the OpenCV wrapper happy:
        self.assert_((mat.asArray() == 2*eye(10)).all())

        (mat, bandIndices) = BandedMatrix(10, -1, 0, 1)  # 10x10 tri-diagonal matrix.
        self.assert_((mat.asArray() == zeros((10,10))).all())
        self.assert_(mat.numNonzeros == 28)
labor = asarray(DataReader('PAYEMS', 'fred',
# Use pandas DataReader to get data from fred
output = asarray(DataReader('GDPC1', 'fred',
                            start=start1, end=end1)).squeeze()
investment = asarray(DataReader('GPDIC96', 'fred',

        self.pt = (-1.725, 2.9, 0.725)
        self.A = asarray([[1.0, 0.0, 1.0], [0.0, 1.0, -4.0]])
        self.b = asarray([-1.0, 0.0])
        self.bounds = ((None, None), (0.1, 2.9), (0.7, None))
    return np.asarray(X, dtype=dtype)

    y = np.asarray(y, dtype=np.int32)
    # the OpenCV wrapper happy:
    # If a out_dir is given, set it:

        return None, value
        return None, None, mesg
    return np.asarray(pressure), aver, mesg

        mat.setRow(9, [1, -1])
        self.assert_((mat.asArray() == self.laplacian).all())

        mat.setValues(bandIndices[2], [1]*9)
        self.assert_((mat.asArray() == self.laplacian).all())
        f.create_dataset("trans", data=np.asarray(self.trans))
        f.create_dataset("rot", data=np.asarray(self.rot))
        f.create_dataset("homs", data=np.asarray(self.homs))
        f.create_dataset("grid", data=np.asarray(self.grid))
        f.close()
    if p==np.inf or p==1:
        """
    x = np.asarray(x)
    y = np.asarray(y)
    """
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # Learn the model. Remember our function returns Python lists,
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
            b = b.asarray()
        a = self.asarray()
        #if isinstance(b, MetaArray):
            #b = b.asarray()
        #return self.asarray() != b
        P = np.asarray(self.P, dtype=np.float)
        dst = blas.dgemv(np.array([P], order='C'), src)
        pixels = np.asarray(dst[:, :2], order='C')
        w = dst[:, 2]
        # Create unit vectors along edges of the view frustum
    x = np.asarray(x)
    sh = list(x.shape)
       Optional mirroring with "mirrored" argument.'''
    points,newRefLine = np.asarray(points), np.asarray(newRefLine)
    dx,dy = (newRefLine[1]-newRefLine[0])
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p==np.inf:
    """Compute the L**p distance between x and y"""
    >>> xy = npy.asarray([[0,0],[0,1],[1,1],[1,0]])
    >>> print avg_speed(xy)

    measures = npy.asarray(measures)

    Homs.append(np.asarray(Hom,dtype=float))
    Rot.append(np.asarray(tmpRot,dtype=float))
    Trans.append(np.asarray(tmpTrans,dtype=float))
    transDist=np.sqrt(tmpTrans[0,0]*tmpTrans[0,0]+tmpTrans[0,1]*tmpTrans[0,1])
    # so we use np.asarray to turn them into NumPy lists to make
            r = np.asarray(r)
            n, = r.shape
    """
        self.assert_((mat.asArray() == eye(10)).all())
        mat = IdentityMatrix(10, 2) # With a 2 coefficient.

    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')
    """
    Y = np.asarray(Y, order='c')
    out = {}
    out['piprads'] = np.asarray(
                            (piprad, PIX_ERR*np.sqrt(2)*np.ones_like(piprad)))
    if img.dtype == np.int32:
        img = np.asarray(np.asfarray(img), np.int32)
                #print "matches found"
        self.data = np.asarray(data)
        self.n, self.m = np.shape(self.data)

    x = np.asarray(x)
                self._data = data.asarray()
            elif isinstance(data, tuple):  ## create empty array with specified shape
        ma = MetaArray._h5py_metaarray.MetaArray(file=fileName)
        self._data = ma.asarray()._getValue()
        self._info = ma._info._getValue()
        """
        x = np.asarray(x)
        if x.shape[-1] != self.m:
        """
        x = np.asarray(x)
        self.assertTrue(np.all(cn.asarray() == np.array([[0.1,0.2],[0.3,0.4]])))
        
        
    def test_asarray(self):
        c = ConfusionMatrix([[1,2],[3,4]])
            extra_img['profile'] = profile
    out['piprads'] = np.asarray((piprads, piprads_err))
    out['vess'] = np.asarray((vess, vess_err))
    out['asps'] = np.asarray((asps, asps_err))
    return out, extra_out
                kp_l = np.asarray(im_left.keypoints)[idx_l]
        cart_coords = misctools.spherical_to_cartesian(
            np.asarray([radius, theta, phi]))
        cart_coords = cart_coords[[0, 2]].T
                desc_l = np.asarray(im_left.descriptors)[idx_l]
    return x - x.mean(*args,**kwds)
       This is the equivalent of the "in" operator when using lists instead of arrays.'''
    arr = np.asarray(arr)
    subarr = np.asarray(subarr)
    if subarr.shape!=arr.shape[1:]:
    tensiondata = {}
    tensiondata['dilation'] = np.asarray((a, da))
    tensiondata['tension'] = np.asarray((tau/1000.0, dtau/1000.0)) #in mN/m
    tensiondata['tensdim'] = ('mN/m',r'$10^{-3}\frac{N}{m}$')
    results['area'] = np.asarray((area,area_err))
            else:
                VI = np.asarray(VI, order='c')
            [VI] = _copy_arrays_if_base_present([VI])
wages = asarray(DataReader('WASCUR', 'fred',
                                start=start1, end=end1)).squeeze()
    assert (Nx == int(Nx)), 'Input "Nx" must be an integer type.'
        yj = asarray(yj)

                          [0.0, 0.0, 1.0, -2.0, -2.0]])
        self.b = asarray([5.0, -3.0])
        self.center = npy.asarray((x0,y0),dtype=float)

        #model center
        self.center = npy.asarray((x0,y0),dtype=float)

    Rot.append(np.asarray(tmpRot,dtype=float))
    Trans.append(np.asarray(tmpTrans,dtype=float))
    transDist=np.sqrt(tmpTrans[0,0]*tmpTrans[0,0]+tmpTrans[0,1]*tmpTrans[0,1])
    # so we use np.asarray to turn them into NumPy lists to make
    # Thanks to Leo Dirac for reporting:
    """
        self.assert_((mat.asArray() == eye(10)).all())
        mat = IdentityMatrix(10, 2) # With a 2 coefficient.
        self.assert_((mat.asArray() == 2*eye(10)).all())

    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')
    """
    Y = np.asarray(Y, order='c')
    is_valid_y(Y, throw=True, name='Y')
    img = asarray(img)
        z = zip(ybad[i]+yj,xbad[i]+xj)
        w = 1.0 / sqrt(asarray(xj)**2 + asarray(yj)**2)

        xj = asarray(xj)
        self.n, self.m = np.shape(self.data)

    x = np.asarray(x)
    m, k = x.shape
    y = np.asarray(y)
        self.pt = (1.0, 1.0, 1.0, 1.0, 1.0)
        self.A = asarray([[1.0, 1.0, 1.0,  1.0,  1.0],
    results['metrics'] = argsdict['metrics']
    tensiondata['dilation'] = np.asarray((alpha,alpha_err))
    tensiondata['tensdim'] = ('mN/m',r'$10^{-3}\frac{N}{m}$')
        x = np.asarray(x)
        if x.shape[-1] != self.m:
        """
        x = np.asarray(x)
        if np.shape(x)[-1] != self.m:
    if img.dtype == np.int32:
        img = np.asarray(np.asfarray(img), np.int32)
    return img, mesg
    out['metrics'] = np.asarray((metrics, np.zeros_like(metrics)))
    return np.asarray(pressures)[stage], None
    from scipy.interpolate import interp1d
    dcb_radiance = asarray(dcb_radiance)

      #convert cvMat to numpy array
      tmpIm = np.asarray(cvFinal)
      
    Hom[2,2]=1
