import os
import gradio as gr
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import ollama  # Import the official local library directly

# --- 1. SETUP, INITIALIZATION & CLIENTS ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize YOLO for Gap Detection
yolo_model = YOLO('best-2.pt') 

# --- 2. MULTI-PLANOGRAM REGISTRY ---
PLANOGRAM_REPOSITORIES = {
   "Aisle 1 - Pet Care & Pet Foods (3x6 Grid)": {
        "matrix": {
            "Row 1 (Top)": ["Pedigree Adult 2kg", "Purina ONE Dry Cat", "Kittens Choice Mix", "Meow Mix Chicken", "Iams Small Breed", "Fancy Feast Salmon"],
            "Row 2 (Middle)": ["Whiskas Jelly Pouches 12x", "Sheba Gold Wet 8x", "Felix Party Mix", "Gourmet Gold Paté", "Dine Delicious", "Dreamies Treats"],
            "Row 3 (Bottom)": ["Hill's Science Diet 4kg", "Royal Canin Size 4kg", "Pro Plan Puppy 12kg", "Bulk Canned Beef", "Bulk Canned Tuna", "Nandog Pet Beds"]
        }
    },
    "Aisle 2 - Snacks & Sodas (5x3 Grid)": {
        "matrix": {
            "Row 1 (Top)": ["Lay's Paprika Blue", "Lay's Paprika Blue", "Lay's Naturel Yellow", "Lay's Naturel Yellow", "Doritos Nacho Cheese", "Doritos Sweet Chili"],
            "Row 2": ["Oreo Original", "Oreo Original", "Oreo Double Cream", "Milka Choco Biscuits", "McVitie's Digestive"],
            "Row 3": ["Pringles Sour Cream Green", "Pringles Sour Cream Green", "Pringles Original Red", "Pringles Original Red", "Pringles Paprika Orange", "Pringles Paprika Orange", "Pringles Hot & Spicy", "Pringles Cheesy Cheese"],
            "Row 4": ["M&M's Peanut Yellow", "M&M's Choco Brown", "Maltesers XL Pack", "Snickers 5-pack", "Mars Bar Multipack"],
            "Row 5 (Bottom)": ["Coca-Cola 1.5L", "Coca-Cola Zero 1.5L", "Fanta Orange 1.5L", "Fanta Exotic 1.5L", "Sprite Lemon 1.5L", "Sprite Zero 1.5L"]
        }
    },
    "Aisle 3 - Baking & Breakfast (4 Flexible Rows)": {
        "matrix": {
            0: ["All-Purpose Flour 2kg", "Granulated Sugar 2kg", "Brown Sugar 1kg", "Dr. Oetker Mix", "Betty Crocker Cake"],
            1: ["Baking Powder", "Yeast Pack", "Vanilla Extract", "Dr. Oetker Icing", "Betty Crocker Frosting", "Mini Marshmallows", "Chocolate Chips"],
            2: ["Quaker Oats Jumbo 1kg", "Weetabix Original", "Cheerios Honey Nut", "Special K Cereal", "Kellogg's Muesli"],
            3: ["Clear Honey Jar 500g", "Manuka Honey 250g", "Bonne Maman Jam", "Smucker's Strawberry", "Nutella Jar 400g", "Lotus Biscoff"]
        }
    },

    "Colruyt - Biscuit Aisle (Flexible Rows)": {
        "matrix": {
            "Row 1 (Top)": ["AH Blue Choco Box", "Boni Red Choco Box", "Delacre Prestige Gold", "Delacre Prestige Gold", "Lu Prince Chocolate"],
            "Row 2 (Bottom)": ["Cote dOr Pasta Red", "Chocopasta White Label", "Blue Tin Spread Pack", "Lotus Speculoos Red", "Boni Vanilla Biscuits", "Choco Snack Pack", "Extra Cookie Box"]
        }
    }
}

# --- 3. ADVANCED LLM REASONING LAYER ---
def query_llm_spatial_reasoning(planogram_template, normalized_gaps, selected_name):
    """
    Local spatial processing engine. Receives pre-calculated math 
    from Python to guarantee 100% accurate product reporting.
    """
    prompt = f"""
You are an expert Retail Space Automation AI analyzing a supermarket shelf via Computer Vision data.
The user has scanned a shelf matching the layout template: "{selected_name}"

Here is the Golden Reference Planogram Matrix:
{planogram_template}

Our vision pipeline has detected gaps and mapped them to their exact target product locations using deterministic spatial math. 

Here are the pre-processed gaps and their verified targets:
{normalized_gaps}

Instructions:
1. Generate an audit report using clean, professional Markdown syntax.
2. For each gap item, create a clean header '### 🚨 GAP [ID] DETECTED'. Specify the row name, and state the 'Missing Product:' exactly as provided in the manifest wrapped in backticks.
3. Conclude with a very brief executive summary statement of compliance.
"""

    try:
        response = ollama.chat(
            model='phi3', 
            messages=[
                {"role": "system", "content": "You are a precise retail compliance auditor. Present the provided data layout cleanly without changing product names."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.0} # Absolute determinism
        )
        return response['message']['content']
    except Exception as e:
        return f"### ❌ Local Spatial Reasoning Engine Error\nPipeline failure. Details: {str(e)}"


# --- 4. DATA PROCESSING & DETERMINISTIC MAPPER ---
def process_shelf_grid_with_ai(gaps, img_height, img_width, selected_planogram_name):
    config = PLANOGRAM_REPOSITORIES[selected_planogram_name]
    matrix = config["matrix"]
    row_keys = list(matrix.keys()) 
    total_rows = len(row_keys)
    
    if len(gaps) == 0:
        return "No gaps detected! Shelf is perfectly stocked.", []

    gap_data_raw = []
    for gap in gaps:
        gx1, gy1, gx2, gy2 = gap.xyxy[0].cpu().numpy()
        xc = (gx1 + gx2) / 2
        yc = (gy1 + gy2) / 2
        
        gap_data_raw.append({
            'xc': xc,
            'yc': yc,
            'coords': (int(gx1), int(gy1), int(gx2), int(gy2))
        })

    normalized_gap_strings = []
    final_ordered_gaps = []
    
    # Sort top-to-bottom so Gap 1 is always the highest up on the physical shelf
    gap_data_raw.sort(key=lambda g: g['yc'])
    
    for idx, gap in enumerate(gap_data_raw, start=1):
        norm_x = gap['xc'] / img_width
        norm_y = gap['yc'] / img_height
        
        # 1. Deterministic Vertical Row Assignment
        row_logical_index = int(norm_y * total_rows)
        row_logical_index = max(0, min(row_logical_index, total_rows - 1))
        assigned_row_name = row_keys[row_logical_index]
        
        # 2. Deterministic Horizontal Array Position Mapping
        row_items = matrix[assigned_row_name]
        row_items_count = len(row_items)
        
        idx_in_row = int(norm_x * row_items_count)
        idx_in_row = max(0, min(idx_in_row, row_items_count - 1))
        
        # Extract the exact product string directly via Python array indexing
        target_missing_product = row_items[idx_in_row]
        
        gap['assigned_id'] = idx
        final_ordered_gaps.append(gap)
        
        # Pass the pre-computed final answer into the LLM payload manifest
        normalized_gap_strings.append(
            f"- Gap ID {idx}: Located on **{assigned_row_name}**.\n"
            f"  Horizontal Scan Percentage: {norm_x:.2f}\n"
            f"  Matched Target Missing Item: {target_missing_product}"
        )
    
    gaps_manifest = "\n".join(normalized_gap_strings)
    matrix_manifest = str(matrix)
    
    ai_audit_report = query_llm_spatial_reasoning(matrix_manifest, gaps_manifest, selected_planogram_name)
    
    return ai_audit_report, final_ordered_gaps


# --- 5. GRADIO INTERFACE EXECUTION ---
def audit_interface(input_img, selected_planogram):
    img_height, img_width, _ = input_img.shape
    
    results = yolo_model.predict(
        input_img, 
        imgsz=(640, 640), 
        conf=0.30,  
        iou=0.20,   
        augment=True      
    )[0]

    # === DEBUG PRINT BLOCK ===
    print("\n--- RAW YOLO DETECTIONS FOUND ---")
    for idx, b in enumerate(results.boxes):
        c = int(b.cls)
        conf = float(b.conf)
        print(f"Detection {idx}: Class={c}, Confidence={conf:.2f}, Box={b.xyxy[0].cpu().numpy().tolist()}")
    print("---------------------------------\n")
    
    gaps = [b for b in results.boxes if int(b.cls) == 0]
    
    report, final_ordered_gaps = process_shelf_grid_with_ai(gaps, img_height, img_width, selected_planogram)
    
    annotated_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    for gap in final_ordered_gaps:
        x1, y1, x2, y2 = gap['coords']
        gap_id = gap['assigned_id']
        
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 165, 255), 4) 
        cv2.putText(annotated_img, f"Gap {gap_id}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        
    final_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return final_img, report


# --- 6. MODERN GRADIO SYSTEM INTERFACE ---
planogram_options = list(PLANOGRAM_REPOSITORIES.keys())

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛒 Planogram Digital Twin Auditor v2")
    gr.Markdown("### Hybrid Architecture: Computer Vision (YOLO) + Local Spatial Reasoning (Ollama - Phi3)")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(label="Upload Shelf Photo", type="numpy")
            aisle_select = gr.Dropdown(
                choices=planogram_options, 
                value=planogram_options[0], 
                label="Select Shelf Layout Template"
            )
            btn = gr.Button("Analyze Planogram Compliance", variant="primary")
        with gr.Column(scale=1):
            img_output = gr.Image(label="Live Space Variance Visualizer")
            text_output = gr.Markdown(label="LLM Compliance Audit Output") 

    btn.click(
        fn=audit_interface, 
        inputs=[img_input, aisle_select], 
        outputs=[img_output, text_output]
    )

if __name__ == "__main__":
    demo.launch()