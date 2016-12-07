import numpy as np
import pyfftw

def reject_outliers(data, m = 2.):
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d/mdev if mdev else 0.
	data[np.where(s>m)] = np.median(data)
	return data

def fourierSolve(imgData, imgGradX, imgGradY):
	imgShape = imgData.shape
	print imgShape
	imgWidth, imgHeight = imgShape
	nodeCount = imgWidth * imgHeight
	imgData = imgData.flatten()
	imgGradX = imgGradX.flatten()
	imgGradY = imgGradY.flatten()

	fftBuff = np.zeros(nodeCount, dtype=np.float)

	# compute two 1D lookup tables for computing the DCT of a 2D Laplacian on the fly
	xRange = np.arange(imgWidth)
	yRange = np.arange(imgHeight)
	ftLapX = 2.0 * np.cos(np.pi * xRange / (imgWidth - 1.0))
	ftLapY = 2.0 * np.cos(np.pi * yRange / (imgHeight - 1.0))

	# # Create a DCT-I plan for, which is its own inverse.
	# fftwf_plan_with_nthreads(NUM_OF_THREADS);
	# fftwf_plan fftPlan;
	# fftPlan = fftwf_plan_r2r_2d(imgHeight, imgWidth,
	# 							 fftBuff, fftBuff,
	# 							 FFTW_REDFT00, FFTW_REDFT00, FFTW_ESTIMATE); #use FFTW_PATIENT when plan can be reused

	nBands = 1
	for iChannel in range(nBands):
		nodeAddr = 0
		pixelAddr = iChannel
		rightPixelAddr = nBands + iChannel
		topPixelAddr = imgWidth * nBands + iChannel

		dcSum = 0.0

		# compute h_hat from u, gx, gy (see equation 48 in the paper), as well as the DC term of u's DCT.

		for y in range(imgHeight):
			for x in range(imgWidth):
				# Compute DC term of u's DCT without computing the whole DCT.
				dcMult = 1.0
				if ((x > 0) and (x < imgWidth  - 1)):
					dcMult *= 2.0
				if ((y > 0) and (y < imgHeight - 1)):
					dcMult *= 2.0
				dcSum += dcMult * imgData[pixelAddr]

				# Subtract g^x_x and g^y_y, with boundary factor of -2.0 to account for boundary reflections implicit in the DCT
				if ((x > 0) and (x < imgWidth - 1)):
					fftBuff[nodeAddr] -= (imgGradX[rightPixelAddr] - imgGradX[pixelAddr])
				else:
					fftBuff[nodeAddr] -= (-2.0 * imgGradX[pixelAddr])

				if ((y > 0) and (y < imgHeight - 1)):
					fftBuff[nodeAddr] -= (imgGradY[topPixelAddr] - imgGradY[pixelAddr])
				else:
					fftBuff[nodeAddr] -= (-2.0 * imgGradY[pixelAddr])

				nodeAddr += 1
				pixelAddr += nBands
				rightPixelAddr += nBands
				topPixelAddr += nBands

		fftBuff = fftBuff.reshape((imgWidth, imgHeight))
		# transform h_hat to H_hat by taking the DCT of h_hat
		fftw = pyfftw.builders.fft2(fftBuff, None, axes=(0, 1))
		fftw.execute()

		fftBuff = fftBuff.flatten()
		# fftBuff = reject_outliers(fftBuff)
		# compute F_hat using H_hat (see equation 29 in the paper)
		nodeAddr = 0
		for y in range(imgHeight):
			for x in range(imgWidth):
				ftLapResponse = ftLapY[y] + ftLapX[x]
				if ftLapResponse == 0.0:
					ftLapResponse = 1
				# print ftLapResponse
				# fftBuff[nodeAddr] /= -ftLapResponse
				nodeAddr += 1

		# Set the DC term of the solution to the value computed above (i.e.,
		# the DC term of imgData).
		# When dataCost = 0 (i.e., there is no data image and the problem
		# becomes pure gradient field integration)
		# then the DC term  of the solution is undefined. So if you want to
		# control the DC of the solution
		# when dataCost = 0 then before calling fourierSolve() set every pixel
		# in 'imgData' to the average value
		# you would like the pixels in the solution to have.
		fftBuff[0] = dcSum

		fftBuff = fftBuff.reshape((imgWidth, imgHeight))

		# transform F_hat to f_hat by taking the inverse DCT of F_hat
		ifftw = pyfftw.builders.ifft2(fftBuff, None, axes=(0, 1))
		ifftw.execute()

		fftBuff = fftBuff.flatten()
		# fftBuff = reject_outliers(fftBuff)

		fftDenom = 4.0 * (imgWidth - 1) * (imgHeight - 1)
		pixelAddr = iChannel
		for iNode in range(nodeCount):
			imgData[pixelAddr] = fftBuff[iNode] / fftDenom
			pixelAddr += nBands

	imgData = reject_outliers(imgData)
	imgData = imgData.reshape(imgShape)
	imgData = imgData - np.min(imgData)
	imgData = imgData * 255.0/np.max(imgData)
	return imgData
