import IP
def bilateral(img, size, sigmaS, sigmaB):
    ''' Apply a midpoint filter with a nxn kernel to 
        a grayscale image.
        @param  image on which filter is to be applied
                size: kernel size
        @return filtered image
'''
    
    l, w = img.shape
    kernelSize = size
    n = (kernelSize-1) // 2
    out = IP.np.zeros((l, w))
    #SUM=IP.np.zeros((l, w))
    SUM=0
    pi=IP.np.pi
    for i in range(n, l-n):
        for j in range(n, w-n):
            kernel = _getKernel(img, kernelSize, i, j)
            #convert the 1D kernel array to NxN kernel
            kernel2D=IP.np.reshape(kernel, (size,size))
            
            for m in range(size):
                for n in range(size):
                    Gs_a=1/(2*pi*sigmaS**2)
                    Gs_b=IP.np.exp(-1/2*(((i-m)**2+(j-n)**2)/sigmaS**2))
                    Gb=1/((2*pi)**(1/2)*sigmaB)*math.exp(-1/2*((img[m,n]-img[i,j])**2/IP.np.power(sigmaB,2)))
                    Gs=Gs_a*Gs_b
                    W=Gs*Gb
                    main=img[m,n]*Gs*Gb
                    SUM=1/W*(SUM+main) 
                    out[i,j]=SUM
    return Gs_b
img=IP.np.zeros((512,512))
bilateral(img, 1, 10, 10)