import numpy as np
from astroML.time_series import generate_damped_RW
import matplotlib.pyplot as plt
import pyfits

#************************
#	parameters	#
#************************

#QSO
lamb = 0.6586		# wavelength in micro-m
Mbh  = 1.3e8 		# mass of the black hole in solar masses
LLE  = 0.1		# L/LE Eddintong ratio
n    = 0.1		# radiative efficiency
zs   = 0.658		# redshift of the source
alf  = 6		# 6 = Schwarszschild black hole or 1 = Kerr black hole
pix  = 1001		# resolution of the kernel in pixels (always odd numbers) app to 800
step = 1.0		# steps for the array usually 1.0 
pad  = [0.0,45.0]		# position angle in degrees	
incd = [0.0,30.0]		# inclination in degrees
norm = 6.15e13		# pixel scale in cm 
sizef = 1.0		#factor to multiply for the disk size
#Map
xcen = [1501,6701]	#coordinates for the central pixel in the map
ycen = [2254,5247]
#DRW
tau = 90		# time scale or relaxation time
ranst = 1		# random seed
fraV  = 0.15		# fraction variability
time  = 120		# in days
#graphic
pox = 10		# coordinates to plot the name of the light curve
poy = 0.15

#conversions
c = 299792458*100 #c in cmeters
days = (60*60*24) #to transform to days
lmin = -(pix/2.0) 		# centerx = totalpixel/2.0
lmax = (pix/2.0)  		# centery = totalpixel/2.0
SF   = (fraV/np.sqrt(2.0))	# Structure funciton at infinity -> default = 0.3

#********************************
#	Damped Random Walk	#
#********************************

t = np.arange(0,time)
y = generate_damped_RW(t, tau=tau, z=zs,random_state=ranst,SFinf=SF) 

#********************************
#	Accretion disk model	#
#********************************

def sb2d(lamb,Mbh,LLE,n,zs,alf,lmin,lmax,step,pa,inc,norm,size):
	R0 = ( 9.7e15 * ((1.0/(1.0+zs))*(lamb))**(4.0/3.0) * (Mbh/10**9)**(2.0/3.0) * (LLE/n)**(1.0/3.0) )
	Rcod = size*R0
	R0p = ( Rcod/norm )
#	G = 6.67e-11 #m**3 kg**-1 s-2
#	c = 299792458 #m/s
#	Rin = ((alf * G * Mbh*1.989e30)/c**2) * 100 #1msun = 1.989e30kg
#	Rinp = R0p/100.0
	Rin = 0.0
	Rinp = Rin/norm
	rx = np.arange(lmin,lmax,step)
	ry = rx[:,np.newaxis]
	rxn = rx*np.cos(pa) - ry*np.sin(pa) #rotation matrix
	ryn = rx*np.sin(pa) + ry*np.cos(pa)
	R = np.sqrt( (rxn**2)/np.cos(inc)**2 + (ryn**2))
	Rinc = rxn*np.tan(inc)
	xi = (R/R0p)**(3./4.) #* ( 1.0 - np.sqrt(Rinp/R)**(-1.0/4.0) )
	G = ( ( xi * np.exp(xi) )/( np.exp(xi) - 1.0 )**2 ) #G(xi) eq9 T&K18
	return G, R0p, Rinc, R

#********************************
#	Magnification Map	#
#********************************

ste = int( (pix-1)/2 ) + 1
mmap = pyfits.open('mapA.fits')[0].data

#************************************************
#	tlag and intensity profile for the DRW	#
#************************************************

LC = [] #file to keep the new values of the intensity
MicroLC = [] #file to keep the new values of the intensity*microlensing
d=0
for j in range(len(xcen)):
	mcut = mmap[(ycen[j]-ste):(ycen[j]+ste-1),(xcen[j]-ste):(xcen[j]+ste-1)]
#	print((ycen[j]-ste),(ycen[j]+ste-1),(xcen[j]-ste),(xcen[j]+ste-1))
	for k in range(len(pad)):
		pa   = (pad[k]*np.pi)/180.0	
		inc  = (incd[k]*np.pi)/180.0	
		G , R0, Rinc, Rlamp = sb2d(lamb,Mbh,LLE,n,zs,alf,lmin,lmax,step,pa,inc,norm,sizef)
		tlag = ( ( (Rlamp - Rinc)/ (c/norm) )*(1.0 + zs)  / (days) ) #in days
		x01 = np.where(tlag == tlag[int(lmin)][int(lmax)])[0][0]#central coordinates
		y01 = np.where(tlag == tlag[int(lmin)][int(lmax)])[0][0]
		R0pix = int(round(R0)) #acretion disk size in a integer pixel
		lagr01 = tlag[x01+(R0pix-1)][y01+(R0pix-1)] #Lag at R0
		d += 1
		print('LC'+str(d),'coordinates: ',xcen[j],ycen[j],'PA:',pad[k],'inc',incd[k])
#		print('lag at R0 ',R0pix,'[pix]:',lagr01,'[days]')
		temp1 = []
		temp2 = []
		for i in range(len(y)):
			I = (G*np.interp(tlag+i,t,y))/sum(sum(G)) #test
			temp1.append(sum(sum(G*mcut*np.interp(tlag+i,t,y)))/sum(sum(G*mcut)))	
			temp2.append(sum(sum(I)))
		LC.append(temp1)
		MicroLC.append(temp2)

ig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex='col',sharey=True,figsize=(20,10))
ax1.plot(t,LC[0],'k-',label='nomicro')
ax1.plot(t,MicroLC[0],'g-',label='micro')
ax1.text(pox, poy, 'LC1', fontsize=10)
ax2.plot(t,LC[2],'k-',t,MicroLC[2],'g-')
ax2.text(pox, poy, 'LC3', fontsize=10)
ax3.plot(t,LC[1],'k-',t,MicroLC[1],'g-')
ax3.text(pox, poy, 'LC2', fontsize=10)
ax4.plot(t,LC[3],'k-',t,MicroLC[3],'g-')
ax4.text(pox, poy, 'LC4', fontsize=10)
ax1.legend(numpoints=1)
plt.subplots_adjust(hspace=0.001)
plt.subplots_adjust(wspace=0.001)
plt.savefig('LC.pdf')
plt.show()

fig, ((ax1,ax2)) = plt.subplots(1,2,figsize=(10, 10))
setmin = min(mmap[(ycen[0]-ste):(ycen[0]+ste-1),(xcen[0]-ste):(xcen[0]+ste-1)].flatten())
setmax = max(mmap[(ycen[0]-ste):(ycen[0]+ste-1),(xcen[0]-ste):(xcen[0]+ste-1)].flatten())

im = ax1.imshow(mmap[(ycen[0]-ste):(ycen[0]+ste-1),(xcen[0]-ste):(xcen[0]+ste-1)],vmin=setmin,vmax=setmax)
ax1.axis([0, pix, 0, pix])
ax1.scatter(ste, ste, s=80, facecolors='none', edgecolors='k')
ax1.set_title('central pixel:'+str(xcen[0])+','+str(ycen[0]))
ax1.axis('off')

ax2.imshow(mmap[(ycen[1]-ste):(ycen[1]+ste-1),(xcen[1]-ste):(xcen[1]+ste-1)],vmin=setmin,vmax=setmax)
ax2.set_title('central pixel:'+str(xcen[1])+','+str(ycen[1]))
ax2.scatter(ste, ste, s=80, facecolors='none', edgecolors='k')
ax2.axis([0, pix, 0, pix])
ax2.axis('off')

plt.subplots_adjust(bottom=0.1, right=0.8, left=0.04,top=0.9,wspace=0.1)
cax = plt.axes([0.83, 0.22, 0.02, 0.56])
plt.colorbar(im,cax=cax)
plt.savefig('coord_map.pdf')
plt.show()

