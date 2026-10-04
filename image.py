import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('image.jpg')

print("data type", img.dtype)
print("dimension", img.ndim)
print("size", img.size)
print("shape", img.shape)

plt.imshow(img)
plt.title("original image")
plt.show()

im2 = img[12:200,12:200,:]
plt.imshow(im2)
plt.title("cropped image")
plt.show()

red = img.copy()
red[:,:,0]=0
plt.imshow(red)
plt.title("no red image")
plt.show()

flipped = np.flip(img, axis = 0)
plt.imshow(flipped)
plt.title("flipped image")
plt.show()

negative = 255 - img

plt.imshow(negative)
plt.title("negative image")
plt.show()

img2 = np.clip(img.astype(np.int16) + 50,0,255).astype(np.uint8)

plt.imshow(img2)
plt.show()

weights = np.array([0.299, 0.587, 0.114])
gray = img[:,:,:3]@weights
plt.imshow(gray, cmap='gray')
plt.title("Grayscale via Dot Product")
plt.show()

bg_replaced = img.copy()
white_mask = (bg_replaced[:, :, 0] > 220) & \
                (bg_replaced[:, :, 1] > 220) & \
                (bg_replaced[:, :, 2] > 220)

bg_replaced[white_mask] = [255, 255, 0]

plt.imshow(bg_replaced)
plt.title("Background Color Replaced")
plt.show()

top_row = np.hstack((img, flipped))
plt.imshow(top_row)
plt.title("collage")
plt.show()


kernel = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

h, w = gray.shape
edges = np.zeros((h, w))

for i in range(1, h - 1):
    for j in range(1, w - 1):
        patch = gray[i-1:i+2, j-1:j+2]
        
        edges[i, j] = np.sum(patch * kernel)

edges = np.abs(edges)

plt.imshow(edges, cmap='gray')
plt.title("Detected Horizontal Edges")
plt.axis('off')
plt.show()