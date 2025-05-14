import random
from PIL import Image, ImageEnhance,ImageFilter,ImageOps



def augment_image(image,object_instance=False):

    if random.random() > 0.2:

        # Random blur (Gaussian blur)
        if random.random() > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))


        # Random saturation adjustment
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.uniform(0.5, 1.2))


        if not object_instance:

            # Random brightness adjustment (0.5 to 1.5)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.5, 1.5))
            # Random contrast adjustment (50% to 150%)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.5, 1.5))
            # Random sharpness (reduce or enhance sharpness)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.5, 2.0))

            # Random horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # Random vertical flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (-30 to 30 degrees)
            image = image.rotate(random.uniform(-30, 30))
           
            # Random grayscale conversion
            if random.random() > 0.5:
                image = ImageOps.grayscale(image)

            # Random edge enhancement
            if random.random() > 0.5:
                image = image.filter(ImageFilter.EDGE_ENHANCE)

            # Random emboss effect
            if random.random() > 0.5:
                image = image.filter(ImageFilter.EMBOSS)

            # Random hue shift (convert to HSV, modify, and convert back)
            hsv = image.convert("HSV")
            hsv_np = list(hsv.split())
            hsv_np[0] = hsv_np[0].point(lambda h: (h + random.randint(-30, 30)) % 256)  # Hue shift
            image = Image.merge("HSV", hsv_np).convert("RGB")

        else:

             # Random brightness adjustment (0.5 to 1.5)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.7, 1.2))
            # Random contrast adjustment (50% to 150%)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.7, 1.2))
            # # Random sharpness (reduce or enhance sharpness)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.7, 1.2))
    # else:
    #     print('As it is','object_instance',object_instance)

    return image


