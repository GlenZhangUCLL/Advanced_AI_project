import gradio as gr
import cv2
import numpy as np
import ollama
import torch
import clip
from PIL import Image
from ultralytics import YOLO

# --- 1. SETUP & MODEL LOADING (Run once) ---
# Using 'mps' for your M3 Pro
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

yolo_model = YOLO('best.pt')
clip_model, preprocess = clip.load("ViT-B/32", device=device)

##candidate_labels = ["Durex box", "Scholl footcare", "Optrex eye drops", "Nicorette pack", "Empty shelf"]
candidate_labels = [
    # Beverages & Soda
    "Coca-Cola bottle", "Pepsi bottle", "Sprite bottle", "Fanta bottle", "Red Bull can", 
    "Monster Energy", "Gatorade", "Evian water", "Volvic water", "Tropicana orange juice",
    # Snacks & Chocolate
    "Lay's chips", "Pringles can", "Doritos bag", "Cheetos", "Oreo cookies", 
    "Kinder Bueno", "KitKat bar", "Snickers bar", "Mars bar", "Cadbury chocolate",
    "Lindt chocolate", "McVitie's biscuits", "Nutella jar", "Ferrero Rocher",
    # Breakfast & Pantry
    "Kellogg's Corn Flakes", "Kellogg's Frosties", "Nestlé Nesquik", "Quaker Oats", 
    "Barilla Pasta", "De Cecco Pasta", "Heinz Tomato Ketchup", "Hellmann's Mayonnaise", 
    "Maggi noodles", "Knorr soup", "Uncle Ben's rice", "Old El Paso",
    # Dairy & Fridge
    "Philadelphia cream cheese", "Lurpak butter", "Activia yogurt", "Müller Corner",
    "Alpro Almond milk", "Oatly milk", "Babybel cheese", "Kerrygold butter",
    # Health, Pharmacy & Personal Care
    "Durex box", "Scholl footcare", "Optrex eye drops", "Nicorette pack", 
    "Colgate toothpaste", "Oral-B toothbrush", "Nivea cream", "Dove soap", 
    "Head & Shoulders", "Pantene shampoo", "Gillette razor", "Listerine mouthwash",
    "Johnson's Baby", "Vaseline jar", "Always pads", "Tampax", "Vicks VapoRub",
    # Household & Cleaning
    "Ariel detergent", "Persil liquid", "Fairy dish soap", "Finish dishwasher", 
    "Domestos bleach", "Mr Muscle", "Lenor softener", "Dettol spray", 
    "Comfort fabric conditioner", "Flash cleaner",
    # General Categories (Fallbacks)
]
text_inputs = clip.tokenize(candidate_labels).to(device)

# --- 2. LOGIC FUNCTIONS ---

def get_neighbors(gap_box, all_items):
    gx1, gy1, gx2, gy2 = gap_box.xyxy[0].cpu().numpy()
    gh = gy2 - gy1  # Height of the gap
    left_neighbor = None
    right_neighbor = None
    
    for item in all_items:
        ix1, iy1, ix2, iy2 = item.xyxy[0].cpu().numpy()
        # Shelf Y-overlap check (approx 50 pixels)
        #if abs(gy1 - iy1) < 50:
        if abs((gy1 + gy2)/2 - (iy1 + iy2)/2) < (gh * 0.5): 
            if ix2 < gx1: # Left
                if left_neighbor is None or ix2 > left_neighbor['coords'][2]:
                    left_neighbor = {'coords': [ix1, iy1, ix2, iy2]}
            elif ix1 > gx2: # Right
                if right_neighbor is None or ix1 < right_neighbor['coords'][0]:
                    right_neighbor = {'coords': [ix1, iy1, ix2, iy2]}
    return left_neighbor, right_neighbor

def identify_brand(pil_img, box_coords):
    crop = pil_img.crop((float(box_coords[0]), float(box_coords[1]), 
                          float(box_coords[2]), float(box_coords[3])))
    image_input = preprocess(crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits_per_image, _ = clip_model(image_input, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return candidate_labels[probs.argmax()]


def get_retail_analysis(left_b, right_b):
    prompt = f"""
    Context: A supermarket shelf gap between '{left_b}' and '{right_b}'.
    Task: Provide a restocking instruction.
    
    Rules:
    1. If the context seems like a misclassification (e.g., 'canned goods' near 'healthcare'), prioritize the more likely category.
    2. If brands match, say: "Restock [Category]."
    3. If brands differ, suggest ONE logical bridge product found in a general store.
    4. Response must be UNDER 6 words. No filler.
    
    Instruction:"""
    try:
        response = ollama.generate(model='llama3.2', prompt=prompt)
        return response['response']
    except Exception as e:
        return f"Reasoning error: {str(e)}"

# --- 3. THE GRADIO WRAPPER ---

def audit_interface(input_img):
    pil_img = Image.fromarray(input_img)
    results = yolo_model.predict(input_img, imgsz=1024, conf=0.12, iou=0.3)[0]
    
    gaps = [b for b in results.boxes if int(b.cls) == 0]
    items = [b for b in results.boxes if int(b.cls) == 1]
    
    annotated_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    report = ""

    for i, gap in enumerate(gaps):
        left, right = get_neighbors(gap, items)
        
        # Fixed: Pass the 'coords' key specifically
        l_brand = identify_brand(pil_img, left['coords']) if left else "Shelf Edge"
        r_brand = identify_brand(pil_img, right['coords']) if right else "Shelf Edge"
        
        verdict = get_retail_analysis(l_brand, r_brand)
        
        # Visualization
        x1, y1, x2, y2 = map(int, gap.xyxy[0].cpu().numpy())
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(annotated_img, f"Gap {i+1}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        
        report += f"### GAP {i+1} Report\n**Context:** {l_brand} ↔ {r_brand}\n**AI Suggestion:** {verdict}\n\n---\n"

    final_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return final_img, report

# --- 4. THE UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛒 Smart Shelf Auditor")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Shelf Photo", type="numpy")
            btn = gr.Button("Analyze Shelf", variant="primary")
        with gr.Column():
            img_output = gr.Image(label="Gap Visualizer")
            text_output = gr.Markdown() 

    btn.click(fn=audit_interface, inputs=img_input, outputs=[img_output, text_output])

demo.launch()