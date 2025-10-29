def extract_overlapping_tiles(image, tile_width, tile_height, overlap):
    tiles = []
    
    # Iterate over the image based on tile_width and tile_height with overlaps
    for start_y in range(0, image.shape[0], tile_height - overlap):
        for start_x in range(0, image.shape[1], tile_width - overlap):
            
            # Calculate the end coordinates with overlap
            end_x = min(start_x + tile_width, image.shape[1])
            end_y = min(start_y + tile_height, image.shape[0])

            # Extract the tile and append it to the tiles list
            tile = image[start_y:end_y, start_x:end_x]
            tiles.append(tile)
    
    return tiles