# Importing libraries
import base64
import os
import fitz

# This function opens a PDF document and converts every page into  a high-resolution image. 
# Each document is saved as a JPG file temporarily. These images are then used for vision-
# based text extraction later on.
def pdf_to_image(path: str, temp_folder: str):
    basename = os.path.basename(path)
    doc = fitz.open(path)
    mat = fitz.Matrix(2, 2)

    # num_pages_to_process = min(2, doc.page_count)
    for page in range(doc.page_count):
        fitz_page = doc.load_page(page)
        save_path = f"{temp_folder}/tmp_{basename}_p{page+1}.jpg"
        pixmap = fitz_page.get_pixmap(matrix=mat)
        pixmap.save(save_path)
    return basename

# This function reads an image file and converts it into a base64 encoded string. 
# This format is required when sending images to the OpenAI API for multimodal processing.
def encode_image(image_path):

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_image