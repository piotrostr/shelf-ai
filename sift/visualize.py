import cv2
import matplotlib.pyplot as plt


def visualize_sift(image_path):
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints
    keypoints = sift.detect(gray, None)

    # Draw keypoints
    img_keypoints = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display results
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Keypoints")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "./sample_image.jpg"
    visualize_sift(image_path)
