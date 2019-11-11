from PIL import Image


img = Image.open("square.png")

img = img.rotate(12)

img.save("square.12.png")
    



