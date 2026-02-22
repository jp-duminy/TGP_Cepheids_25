"""
author: @mimi

TGP Cepheids 25-26

This defines the 'zero-point' orientation for each cepheid's images, taken as the FLIPSTAT value
from the .fits headers on what we define as our reference night, which for us is 2025-10-06 because
we took our best and most comprehensive data that night. This enables the photometry pipeline to autorotate
pixel guesses for reference stars when the image is flipped.

Night-to-night, the images are also translated, so you need to go one beyond the simple FLIPSTAT 180
degree rotation. We did this by including a translation: you anchor the zero-point of the image as the 
cepheid on the reference night, then since we have a cepheid pixel guess for each night, you can compute
the overall pixel translation of the image. This is handled in the photometry functions.

"""

cepheid_orientation_catalogue = {
    "01": "East",
    "02": "East",
    "03": "East",
    "04": "East",
    "05": "West",
    "06": "West",
    "07": "West",
    "08": "West",
    "09": "West",
    "10": "West",
    "11": "West",
}
