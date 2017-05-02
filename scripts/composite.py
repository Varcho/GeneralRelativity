# Takes an input image folder, observer sphere image, wormhole
# sphere image, and accretion image, and remaps the colors to 
# the rendered output
# Copyright (C) Bill Varcho

from scipy import misc
from scipy.interpolate import interp2d
from numpy import pi
import numpy as np

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import datetime

REMAP = 'out/medium/outimg_00000.png'
LOWER_SPHERE = 'beach.jpg'
UPPER_SPHERE = 'milkyway.jpg'
ACCRETION_DISK = 'accum_ring.png'

remap = misc.imread(REMAP)
upper = misc.imread(UPPER_SPHERE)
lower = misc.imread(LOWER_SPHERE)
accr = misc.imread(ACCRETION_DISK)

uh,uw,_ = upper.shape
lh,lw,_ = lower.shape
ah,aw,_ = accr.shape

print(upper.shape)
print(lower.shape)
print(accr.shape)

ur = interp2d(np.arange(0,uw),np.arange(0,uh),upper[:,:,0])
ug = interp2d(np.arange(0,uw),np.arange(0,uh),upper[:,:,1])
ub = interp2d(np.arange(0,uw),np.arange(0,uh),upper[:,:,2])

lr = interp2d(np.arange(0,lw),np.arange(0,lh),lower[:,:,0])
lg = interp2d(np.arange(0,lw),np.arange(0,lh),lower[:,:,1])
lb = interp2d(np.arange(0,lw),np.arange(0,lh),lower[:,:,2])

# ar = interp2d(np.arange(0,aw),np.arange(0,ah),lower[:,:,0])
# ag = interp2d(np.arange(0,aw),np.arange(0,ah),lower[:,:,1])
# ab = interp2d(np.arange(0,aw),np.arange(0,ah),lower[:,:,2])

def interpolate(theta,phi,col,phi_off=.5):
  phi,theta,l = col[0],col[1],col[2]
  phi = (phi/255.0 + phi_off) % 1.0
  theta = (theta/255.0) % 1.0
  ix,iy = uw*phi,uh*theta
  if l > 0.6*255:
    rr = ur(ix,iy)
    rg = ug(ix,iy)
    rb = ub(ix,iy)
    return np.array([rr,rg,rb])
  elif l > .3*255:
    # rr = ar(ix,iy)
    # rg = ag(ix,iy)
    # rb = ab(ix,iy)
    return np.array([0,180,210])
  else:
    rr = lr(ix,iy)
    rg = lg(ix,iy)
    rb = lb(ix,iy)
    return np.array([rr,rg,rb])


height,width,depth = remap.shape
# for INDEX,phi_offset in enumerate(np.linspace(0,1.0,1)):
INDEX,phi_offset = 0,0
out_img = np.zeros(remap.shape)
for i,theta in enumerate(np.linspace(0, pi, height)):
	for j,phi in enumerate(np.linspace(0, 2*pi, width)):
	  res = interpolate(theta,phi,remap[i,j],phi_offset)
	  out_img[i,j,0] = res[0]
	  out_img[i,j,1] = res[1]
	  out_img[i,j,2] = res[2]

# water mark the image
img = Image.fromarray(np.uint8(out_img))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("/Library/Fonts/arial.ttf", 12)
draw.text((0, 355),'bill varcho',(255,255,255),font=font)
draw.text((0, 365),str(datetime.date.today()),(255,255,255),font=font)
draw.text((0, 375),'ellis wormhole',(255,255,255),font=font)
draw.text((0, 385),'phi_off: ' + str(phi_offset*2*pi),(255,255,255),font=font)
img.save('composited/a_' + str(INDEX) + '.png')