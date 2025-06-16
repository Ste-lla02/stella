import os
from PIL import Image, ImageDraw
from src.utils.configuration import Configuration

# Definisci la funzione per ritagliare un'immagine in base ai punti del poligono
def crop_image_with_polygon(image, image_name):
    configuration = Configuration()
    points = configuration.get('areaofinterest')
    output_dir = configuration.get('croppedfolder')
    # Carica l'immagine
    # Creare una maschera con lo stesso formato e dimensione dell'immagine
    mask = Image.new("L", image.size, 0)  # 'L' significa scala di grigi (0-255)
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=255)  # Disegna il poligono e riempilo con bianco (255)
    # Applica la maschera all_400'immagine originale
    result = Image.new("RGBA", image.size)  # Creare un'immagine con canale alpha
    result.paste(image, mask=mask)  # Applica la maschera per conservare solo il poligono
    # Ritaglia l'immagine al bounding box del poligono per risparmiare spazio
    x_coords, y_coords = zip(*points)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    result_cropped = result.crop((x_min, y_min, x_max, y_max))
    # Salva l'immagine ritagliata
    output_path = os.path.join(output_dir, f"{image_name}_cropped.png")
    result_cropped.save(output_path)
    print(f"Immagine salvata: {output_path}")
    return result_cropped
