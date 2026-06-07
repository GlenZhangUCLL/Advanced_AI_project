import os
import gradio as gr
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import ollama  # Import the official local library directly
from planograms import PLANOGRAM_REPOSITORIES

# --- 1. SETUP & INITIALIZATION --- 
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize YOLO for Gap Detection
yolo_model = YOLO('best-2.pt') 

# --- 2. MULTI-PLANOGRAM REGISTRY ---
# --- IMPORT DUMMY DATABASE VIA planograms.py ---


# --- 3. LLM REASONING LAYER ---
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

    # === CALIBRATION STEP: DEFINE ACTIVE PRODUCT ZONE ===
    # Adjust these percentages to perfectly frame where products start and end
    SHELF_X_MIN = 0.12  # Ignoring the left metal pillar/aisle space (starts at 12% width)
    SHELF_X_MAX = 0.95  # Ignoring the rightmost edge frame (ends at 95% width)
    
    # Vertical clipping to account for ceiling/floor dead space or aisle signage
    SHELF_Y_MIN = 0.15  # Adjust this if top shelf starts lower down the image frame
    SHELF_Y_MAX = 0.88  # Adjust this if bottom shelf stops before the image frame base
    
    active_x1 = img_width * SHELF_X_MIN
    active_x2 = img_width * SHELF_X_MAX
    active_shelf_width = active_x2 - active_x1

    # Calculate physical bounding box for vertical shelf space
    active_y1 = img_height * SHELF_Y_MIN
    active_y2 = img_height * SHELF_Y_MAX
    active_shelf_height = active_y2 - active_y1

    gap_data_raw = []
    
    for gap in gaps:
        gx1, gy1, gx2, gy2 = gap.xyxy[0].cpu().numpy()
        box_w = gx2 - gx1
        xc = (gx1 + gx2) / 2
        yc = (gy1 + gy2) / 2
        
        # Estimate row item counts to calculate expected slot width dynamically
        # UPDATED: Evaluated against active shelf height window
        norm_y = (yc - active_y1) / active_shelf_height
        norm_y = max(0.0, min(norm_y, 0.999))
        
        row_logical_idx = max(0, min(int(norm_y * total_rows), total_rows - 1))
        assigned_row_name = row_keys[row_logical_idx]
        row_items_count = len(matrix[assigned_row_name])
        
        # Base slot width on the active shelf area, not the raw canvas width
        expected_slot_width = active_shelf_width / row_items_count
        
        # If the bounding box is wide enough to contain 2 adjacent empty spaces
        if box_w > (expected_slot_width * 1.5):
            half_w = box_w / 2
            # Programmatically generate Left Gap Component
            gap_data_raw.append({
                'xc': gx1 + (half_w / 2),
                'yc': yc,
                'coords': (int(gx1), int(gy1), int(gx1 + half_w), int(gy2))
            })
            # Programmatically generate Right Gap Component
            gap_data_raw.append({
                'xc': gx1 + half_w + (half_w / 2),
                'yc': yc,
                'coords': (int(gx1 + half_w), int(gy1), int(gx2), int(gy2))
            })
        else:
            gap_data_raw.append({
                'xc': xc,
                'yc': yc,
                'coords': (int(gx1), int(gy1), int(gx2), int(gy2))
            })

    # Two-pass row sorting
    gap_data_raw.sort(key=lambda g: g['yc'])
    row_threshold = img_height * 0.10
    grouped_rows = []
    
    for gap in gap_data_raw:
        placed = False
        for physical_row in grouped_rows:
            if abs(gap['yc'] - physical_row[0]['yc']) < row_threshold:
                physical_row.append(gap)
                placed = True
                break
        if not placed:
            grouped_rows.append([gap])
            
    grouped_rows.sort(key=lambda r: sum(g['yc'] for g in r) / len(r))
    
    for physical_row in grouped_rows:
        physical_row.sort(key=lambda g: g['xc'])
        
    final_ordered_gaps = []
    normalized_gap_strings = []
    global_idx = 1
    
    for physical_row in grouped_rows:
        for gap in physical_row:
            # === NORMALIZE Y AGAINST ACTIVE ZONE ===
            norm_y = (gap['yc'] - active_y1) / active_shelf_height
            norm_y = max(0.0, min(norm_y, 0.999)) # Clamp between 0.0 and 1.0 boundary
            
            # === NORMALIZE X AGAINST ACTIVE ZONE ===
            norm_x = (gap['xc'] - active_x1) / active_shelf_width
            norm_x = max(0.0, min(norm_x, 0.999)) # Clamp between 0.0 and 1.0 boundary
            
            row_logical_index = max(0, min(int(norm_y * total_rows), total_rows - 1))
            assigned_row_name = row_keys[row_logical_index]
            
            row_items = matrix[assigned_row_name]
            row_items_count = len(row_items)
            
            idx_in_row = max(0, min(int(norm_x * row_items_count), row_items_count - 1))
            target_missing_product = row_items[idx_in_row]
            
            gap['assigned_id'] = global_idx
            final_ordered_gaps.append(gap)
            
            normalized_gap_strings.append(
                f"- Gap ID {global_idx}: Located on **{assigned_row_name}**.\n"
                f"  Horizontal Scan Percentage: {norm_x:.2f}\n"
                f"  Vertical Scan Percentage: {norm_y:.2f}\n"
                f"  Matched Target Missing Item: {target_missing_product}"
            )
            global_idx += 1
    
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