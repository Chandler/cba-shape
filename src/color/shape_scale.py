import numpy as np

def shape_scale_to_color(v):
	green       = (0,1,0)
	cyan        = (0,1,0.5)
	blue        = (0,1,1)
	pale_blue   = (0.5,1,1)
	white       = (1,1,1)
	pale_yellow = (1,1,0.5)
	yellow      = (1,1,0)
	orange      = (1,0.5,0)
	red         = (1,0,0)

	if -1 <= v < -7.0/8.0:
		return green       # cup
	if  -7.0/8.0 <= v < -5.0/8.0:
		return cyan        # dome
	if -5.0/8.0 <= v < -3.0/8.0:
		return blue        # rut
	if -3.0/8.0 <= v < -1.0/8.0:
		return pale_blue   # saddle rut
	if -1.0/8.0 <= v < 1.0/8.0:
		return white       # saddle
	if 1.0/8.0 <= v < 3.0/8.0:
		return pale_yellow # saddle ridge
	if 3.0/8.0 <= v < 5.0/8.0:
		return yellow      # ridge
	if 5.0/8.0 <= v < 7.0/8.0:
		return orange      # dome
	if 7.0/8.0 <= v <= 1.0:
		return red         # cap


