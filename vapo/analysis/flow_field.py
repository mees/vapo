import affordance.utils.flowlib as flowlib
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (7, 7)

n_values = 16
factor = 20
x_color, y_color = np.meshgrid(np.linspace(-1, 1, n_values * factor), np.linspace(-1, 1, n_values * factor))
x_arrow, y_arrow = np.meshgrid(np.linspace(-1, 1, n_values), np.linspace(-1, 1, n_values))
u = x_arrow
v = y_arrow

flow = np.stack([y_color, x_color]).transpose((1, 2, 0))
flow_img = flowlib.flow_to_image(flow)[:, :, ::-1]

# Create Circle
im_res = n_values * factor
w = im_res // 2
circle = cv2.circle(np.zeros_like(flow_img), (w, w), w, (255, 255, 255), -1)
white = np.ones_like(circle) * 255
flow_img[circle == 0] = flow_img[circle == 0] * 0.6

# Draw lines
line_thickness = 2
flow_img = cv2.line(flow_img.copy(), (w, 0), (w, im_res), (0, 0, 0))
flow_img = cv2.line(flow_img, (0, w), (im_res, w), (0, 0, 0))

widths = np.linspace(0, 3, x_arrow.size)
plt.quiver(x_arrow, y_arrow, u, v, headwidth=8, linewidths=widths)
plt.axis("off")

# cv2.imshow('flow', flow_img)
# cv2.waitKey(0)
# plt.show()
save_folder = "/mnt/484A4CAB4A4C9798/Users/Jessica/Documents/Maestria_Local/Proyecto_hdd/VAPO/Images"

# Save images
cv2.imwrite("%s/color.png" % save_folder, flow_img)
plt.savefig("%s/flow.png" % save_folder, bbox_inches="tight", pad_inches=0)
