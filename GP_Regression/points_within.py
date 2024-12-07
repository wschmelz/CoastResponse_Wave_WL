import numpy
from numba import njit

@njit
def points_mask(x_coords, y_coords, comp_bound):

	mask = numpy.empty_like(x_coords, dtype=numpy.bool_)
	print('')
	for idx in range(x_coords.shape[0]):
		x = x_coords[idx]
		y = y_coords[idx]
		mask[idx] = single_point_within(x, y, comp_bound)
		if (idx+1) % 5000 == 0:
			print("\rcompleted", idx+1, "of", x_coords.shape[0])
	return mask

@njit
def single_point_within(x, y, comp_bound):
    
    tx1 = x
    ty1 = y
    i = 0

    bx2_bx1 = 1000000.0
    by2_by1 = 0.00001

    basevertang = numpy.arctan2(by2_by1, bx2_bx1)

    tx2 = tx1 + 1000000.0 * numpy.cos(basevertang)
    ty2 = ty1 + 1000000.0 * numpy.sin(basevertang)

    A11 = ty2 - ty1
    A12 = -1 * (tx2 - tx1)
    b11 = -1 * ((tx2 * ty1) - (tx1 * ty2))

    for t in range(comp_bound.shape[0]):
        if t == comp_bound.shape[0] - 1:
            shox1, shoy1 = comp_bound[t]
            shox2, shoy2 = comp_bound[0]
        else:
            shox1, shoy1 = comp_bound[t]
            shox2, shoy2 = comp_bound[t + 1]

        if shoy2 - shoy1 == 0:
            shoy2 = shoy1 - .0000000002
        if shox2 - shox1 == 0:
            shox2 = shox1 + .000000001

        A21 = shoy1 - shoy2
        A22 = -1 * (shox1 - shox2)
        b12 = -1 * ((shox1 * shoy2) - (shox2 * shoy1))
        A = numpy.array([[A11, A12], [A21, A22]])
        b = numpy.array([[b11], [b12]])

        solution = numpy.linalg.solve(A, b)

        solx = solution[0][0]
        soly = solution[1][0]

        dotsolsho = ((shox2 - shox1) * (solx - shox1) + (shoy2 - shoy1) * (soly - shoy1))
        dotsho2 = (shox2 - shox1) ** 2 + (shoy2 - shoy1) ** 2

        if dotsolsho >= 0 and dotsho2 >= dotsolsho and solx > tx1:
            i += 1

    return i % 2 > 0