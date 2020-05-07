import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


MODE = 'dark' # either 'light' or 'dark'
if MODE == 'dark':
	plt.style.use('dark_background')
else:
	plt.style.use('seaborn')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


np.random.seed(13)

'''
DATA GENERATION
'''

# eigens
A = np.random.rand(2,2)
eigenvalues, eigenvectors = eig(A)
eigen_log = []

# basis
basis = np.eye(2)
basis_log = []

# random objects
X = -2*np.random.rand(2, 15) + 1
X_log = []

x_grid = np.arange(-8, 8, 1)
y_grid = np.arange(-8, 8, 1)
x_lines = [([x]*2, [-10, 10]) for x in x_grid]
y_lines = [([-10, 10], [y]*2) for y in y_grid]
x_lines_log = []
y_lines_log = []

for i in np.linspace(0, 1.05, 21):
	inter = np.eye(2) + i*(A - np.eye(2))
	eigen_log.append(np.dot(inter, eigenvectors))
	basis_log.append(np.dot(inter, basis))
	X_log.append(np.dot(inter, X))
	x_lines_log.append([np.dot(inter, line) for line in x_lines])
	y_lines_log.append([np.dot(inter, line) for line in y_lines])


'''
ANIMATION
'''

# preparing figure and axes
fig, axs = plt.subplots(1, 1, figsize = (4.5, 4.5))
plt.subplots_adjust(top = .8, bottom = .1, wspace = .4)

# function for redrawing at each frame
def redraw(frame):

	if MODE == 'dark':
		axs.grid(color = 'dimgray')
	axs.set_axisbelow(True)
	axs.set_xlim(-2.5, 2)
	axs.set_ylim(-1.9, 2.3)
	axs.set_xticks(np.arange(-2, 2, 1), minor = False)
	axs.set_yticks(np.arange(-1, 2.5, 1), minor = False)
	axs.set_aspect("equal")

	# transformed grid
	for line in x_lines_log[frame - 20]:
		axs.plot(line[0], line[1], alpha = .2, lw = 1, color = colors[0])
	for line in y_lines_log[frame - 20]:
		axs.plot(line[0], line[1], alpha = .2, lw = 1, color = colors[0])

	offset = .05
	for i in [0, 1]:
		# eigenvectors
		if MODE == 'light':
			axs.arrow(0, 0, eigen_log[frame - 20][:, i][0], eigen_log[frame - 20][:, i][1],
					color = colors[i + 2], width = .015, head_width = .05)
		else:
			axs.arrow(0, 0, eigen_log[frame - 20][:, i][0], eigen_log[frame - 20][:, i][1],
					color = colors[i + 2], head_width = .05)
		axs.text(eigen_log[frame - 20][:, i][0] - offset, eigen_log[frame - 20][:, i][1] + 
				offset, 'Eigenvector {}'.format(i + 1), color = colors[i + 2], 
				size = 'x-small', weight = 'bold', ha = 'right')

		# transformed basis
		if MODE == 'light':
			axs.arrow(0, 0, basis_log[frame - 20][:, i][0], basis_log[frame - 20][:, i][1],
					color = colors[0], width = .015, head_width = .05)
		else:
			axs.arrow(0, 0, basis_log[frame - 20][:, i][0], basis_log[frame - 20][:, i][1],
					color = colors[0], head_width = .05)
		axs.text(basis_log[frame - 20][:, i][0] - offset, basis_log[frame - 20][:, i][1] + 
				offset, 'Standard unit vector', color = colors[0], size = 'x-small', 
				weight = 'bold', ha = 'right')

	# transformed objects
	axs.scatter(X_log[frame - 20][0], X_log[frame - 20][1], alpha = .5, s = 25, 
			color = 'purple')


# beginning with original grid and objects
redraw(20)

# animation update function
def show_animation(frame):

	plt.suptitle('Linear Transformation and Eigenvectors', y = .9, size = 'x-large')
	# 20 to 40: transformation
	if 19 < frame < 41:
		axs.cla()
		redraw(frame)
	# 41 to 60: static
	# 61 to 81: back-transformation
	if 60 < frame < 82:
		axs.cla()
		redraw(80 - frame)

anim = FuncAnimation(fig, show_animation, interval = 40, frames = 81)

anim.save('eigen.gif', writer = 'imagemagick', dpi = 200)
