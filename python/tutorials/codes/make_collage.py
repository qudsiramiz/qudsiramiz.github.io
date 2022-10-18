import matplotlib.pyplot as plt
import glob as glob


# Define a function to make a collage of images
def make_collage(images, n_rows, n_cols, figsize=(10, 10)):
    # Create a figure
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

    try:
        # Loop through the images
        for i in range(n_rows):
            for j in range(n_cols):
                # Plot the image
                ax[i, j].imshow(images[i * n_cols + j], cmap='gray')
    except Exception:
        pass

    # Remove the axis labels
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])

    # Show the figure
    plt.show()


# Define a function to load images
def load_images(path):
    # Get the list of files
    files = glob.glob(path)

    # Create an empty list to store the images
    images = []

    # Loop through the files
    for file in files:
        # Load the image
        image = plt.imread(file)

        # Add the image to the list
        images.append(image)

    # Return the list of images
    return images


# Load the images
images = load_images('../images/*.jpg')

# Make a collage of the images
make_collage(images, 4, 3, figsize=(10, 7))
