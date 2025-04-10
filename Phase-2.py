# Apply Canny Edge Detection
edges = cv2.Canny(blur, 50, 150)

# Invert edges for white background and black lines
inverted_edges = cv2.bitwise_not(edges)

# === Step 3: Resize the image for robot's drawing area (adjust size) ===
resized = cv2.resize(inverted_edges, (300, 300))  # 300x300 can be changed based on robot arm

# === Step 4: Convert to binary image ===
binary = resized > 0  # Boolean image

# === Step 5: Skeletonize the binary image (thin line drawing) ===
skeleton = skeletonize(binary)
skeleton_img = (skeleton * 255).astype(np.uint8)

# === Step 6: Extract drawing coordinates ===
points = np.column_stack(np.where(skeleton_img > 0))

# === Step 7: Display results ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(inverted_edges, cmap='gray')
plt.title('Inverted Edge Detection')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(skeleton_img, cmap='gray')
plt.title('Final Sketch (Skeleton)')
plt.axis('off')

plt.show()

# === Step 8: Save outputs (optional) ===
cv2.imwrite("inverted_sketch.png", inverted_edges)
cv2.imwrite("resized_sketch.png", resized)
cv2.imwrite("skeleton_sketch.png", skeleton_img)

# === Step 9: Show first few points for debugging ===
print("First 10 sketch coordinates:")
print(points[:10])
