import fitz  # PyMuPDF

def slice_long_pdf(input_path, output_path, page_width=595, page_height=842):
    """
    Slices a long single-page PDF into multiple standard-sized pages.
    Default sizes are A4 (595 x 842 points).
    """
    doc = fitz.open(input_path)
    long_page = doc[0]  # Assuming the first page is the long one
    
    # Create a new PDF to hold the slices
    new_doc = fitz.open()
    
    # Get the total height and width of the messy page
    total_width = long_page.rect.width
    total_height = long_page.rect.height
    
    # Determine horizontal offset to "center" or "clip" the content
    # This addresses the 'extra space on the edges' you mentioned
    left_margin = (total_width - page_width) / 2 if total_width > page_width else 0
    
    current_top = 0
    while current_top < total_height:
        # Define the area to 'zoom in' on for the current page
        # Rect(x0, y0, x1, y1)
        crop_rect = fitz.Rect(
            left_margin, 
            current_top, 
            left_margin + page_width, 
            current_top + page_height
        )
        
        # Add a new page to the destination PDF
        new_page = new_doc.new_page(width=page_width, height=page_height)
        
        # Place the cropped section onto the new page
        new_page.show_pdf_page(new_page.rect, doc, 0, clip=crop_rect)
        
        current_top += page_height

    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    print(f"Success! Sliced PDF saved as: {output_path}")

# Run the script
slice_long_pdf("README.pdf", "fixed_document.pdf")