import os
import gradio as gr
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import ollama  # Import the official local library directly
import json

# --- 1. SETUP & INITIALIZATION --- 
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize YOLO for Gap Detection
yolo_model = YOLO('best-2.pt') 

# --- 2. MULTI-PLANOGRAM REGISTRY ---
# --- IMPORT DUMMY DATABASE VIA planograms.py ---
# Load the planogram registry dynamically from JSON
def load_planograms():
    with open("planograms.json", "r") as f:  # Changed "file" to "r"
        return json.load(f)
    
PLANOGRAM_REPOSITORIES = load_planograms()

# --- 3. LLM REASONING LAYER ---
def query_llm_executive_summary(gap_report_body, selected_name):
    """
    Extracts a clean, unique list of missing product names only.
    Strictly prevents model text-bleeding or next-instruction hallucinations.
    """
    prompt = f"""
Input Data for "{selected_name}":
{gap_report_body}

Task: Extract all missing product names from the input data above. Duplicate items must be merged so each product name appears only once. Output the final names as a clean bulleted list using hyphens. Do not include quantities, notes, or subsequent instruction labels.

Response:
- """

    try:
        response = ollama.chat(
            model='phi3', 
            messages=[
                {
                    "role": "system", 
                    "content": "You are a raw text extraction filter. Your only function is to output a unique list of item names based on provided text. Never append text, headers, numbers, or subsequent instructions after the list completes. End the response immediately when the list is finished."
                },
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.0,
                "stop": ["\n\n\n", "Instruction", "Task 2"]
            }
        )
        
        # --- CLEAN UP THE OUTPUT TO FIX INDENTATION ---
        raw_content = response['message']['content'].strip()
        
        # Strip away any accidental leading bullets/spaces the model generated anyway
        if raw_content.startswith(("-", "*", "•")):
            # Remove the character and clean up remaining whitespace
            raw_content = raw_content[1:].strip() 
            
        # Re-attach a perfectly clean Markdown bullet point
        return "- " + raw_content
        
    except Exception as e:
        return f"### ❌ Local Executive Summary Engine Error\nPipeline failure. Details: {str(e)}"

# --- 4. DATA PROCESSING & DETERMINISTIC MAPPER ---
def process_shelf_grid_with_ai(gaps, img_height, img_width, selected_planogram_name):
    config = PLANOGRAM_REPOSITORIES[selected_planogram_name]
    matrix = config["matrix"]
    row_keys = list(matrix.keys()) 
    total_rows = len(row_keys)
    
    if len(gaps) == 0:
        return "No gaps detected! Shelf is perfectly stocked.", []

    # === CALIBRATION STEP: DEFINE ACTIVE PRODUCT ZONE ===
    SHELF_X_MIN = 0.12  # Ignoring the left metal pillar/aisle space (starts at 12% width)
    SHELF_X_MAX = 0.95  # Ignoring the rightmost edge frame (ends at 95% width)
    
    SHELF_Y_MIN = 0.15  # Adjust this if top shelf starts lower down the image frame
    SHELF_Y_MAX = 0.88  # Adjust this if bottom shelf stops before the image frame base
    
    active_x1 = img_width * SHELF_X_MIN
    active_x2 = img_width * SHELF_X_MAX
    active_shelf_width = active_x2 - active_x1

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
        norm_y = (yc - active_y1) / active_shelf_height
        norm_y = max(0.0, min(norm_y, 0.999))
        
        row_logical_idx = max(0, min(int(norm_y * total_rows), total_rows - 1))
        assigned_row_name = row_keys[row_logical_idx]
        row_items_count = len(matrix[assigned_row_name])
        
        # Base slot width on the active shelf area, not the raw canvas width
        expected_slot_width = active_shelf_width / row_items_count
        
        # === DYNAMIC CONTEXT-AWARE GAP SPLITTING ===
        # Calculate exactly how many sequential product slots this single bounding box spans
        num_slots_spanned = max(1, round(box_w / expected_slot_width))
        
        if num_slots_spanned > 1:
            # Segment the large bounding box into proportional slices
            slice_w = box_w / num_slots_spanned
            
            for i in range(num_slots_spanned):
                seg_x1 = gx1 + (i * slice_w)
                seg_x2 = seg_x1 + slice_w
                seg_xc = (seg_x1 + seg_x2) / 2
                
                gap_data_raw.append({
                    'xc': seg_xc,
                    'yc': yc,
                    'coords': (int(seg_x1), int(gy1), int(seg_x2), int(gy2))
                })
        else:
            # Standard single item gap baseline
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
    python_markdown_gaps = []
    global_idx = 1
    
    for physical_row in grouped_rows:
        for gap in physical_row:
            # === NORMALIZE Y AGAINST ACTIVE ZONE ===
            norm_y = (gap['yc'] - active_y1) / active_shelf_height
            norm_y = max(0.0, min(norm_y, 0.999))
            
            # === NORMALIZE X AGAINST ACTIVE ZONE ===
            norm_x = (gap['xc'] - active_x1) / active_shelf_width
            norm_x = max(0.0, min(norm_x, 0.999))
            
            row_logical_index = max(0, min(int(norm_y * total_rows), total_rows - 1))
            assigned_row_name = row_keys[row_logical_index]
            
            row_items = matrix[assigned_row_name]
            row_items_count = len(row_items)
            
            idx_in_row = max(0, min(int(norm_x * row_items_count), row_items_count - 1))
            target_missing_product = row_items[idx_in_row]
            
            gap['assigned_id'] = global_idx
            final_ordered_gaps.append(gap)
            
            # === DYNAMIC 1-INDEXING FOR NUMERIC ROWS ===
            # If the JSON key is a raw digit (like "0"), convert it to a readable 1-indexed number
            if str(assigned_row_name).isdigit():
                display_row_name = str(int(assigned_row_name) + 1)
            else:
                display_row_name = assigned_row_name
            
            # Build the mechanical list using the newly adjusted display name
            python_markdown_gaps.append(
                f"### 🚨 GAP {global_idx} DETECTED\n"
                f"- **Row Name**: {display_row_name}\n"
                f"- **Missing Product**: `{target_missing_product}`\n"
            )
            global_idx += 1
    
    # Combine the individual gap items into a single string body
    gap_report_body = "\n".join(python_markdown_gaps)
    
    # Pass the compiled list to the LLM for a high-quality summary statement
    executive_summary = query_llm_executive_summary(gap_report_body, selected_planogram_name)
    
    # Structure the final comprehensive document returned to the Gradio UI
    full_ai_audit_report = (
        f"# Retail Space Automation AI Audit Report - Supermarket Shelf Analysis\n\n"
        f"## Gap Detection Summary:\n\n{gap_report_body}\n"
        f"## Executive Summary Statement of Compliance:\n{executive_summary}"
    )
    
    return full_ai_audit_report, final_ordered_gaps

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
    gr.Markdown("# 🛒 Planogram Digital Twin Auditor")
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