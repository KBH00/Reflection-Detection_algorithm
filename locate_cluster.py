import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_adaptive_roi(video_width, video_height):
    """
    Calculate an adaptive ROI for 4K based on fixed ratios.
    For example, suppose we want:
      y: [600, 1800] and x: [1300, 2500]
    These numbers are relative to a 4K base (3840x2160) so we compute:
      y_ratio = [600/2160, 1800/2160]
      x_ratio = [1300/3840, 2500/3840]
    Here we use modified numbers:
      y_ratio = [0/2160, 2160/2160] and x_ratio = [150/3840, 2900/3840]
    Returns two lists: y_roi and x_roi.
    """
    base_width = 3840
    base_height = 2160

    y_ratio = [0 / base_height, 2160 / base_height]
    x_ratio = [150 / base_width, 2700 / base_width]

    y_roi = [int(y_ratio[0] * video_height), int(y_ratio[1] * video_height)]
    x_roi = [int(x_ratio[0] * video_width), int(x_ratio[1] * video_width)]
    
    return y_roi, x_roi

def non_max_suppression(clusters, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to clusters."""
    if not clusters:
        return []
    
    # Sort clusters by sum_diff (confidence)
    clusters = sorted(clusters, key=lambda x: x['sum_diff'], reverse=True)
    kept_clusters = []
    
    while clusters:
        best_cluster = clusters.pop(0)
        kept_clusters.append(best_cluster)
        
        non_overlapping = []
        for cluster in clusters:
            if calculate_iou(best_cluster['bbox'], cluster['bbox']) <= iou_threshold:
                non_overlapping.append(cluster)
        clusters = non_overlapping
    
    return kept_clusters

def process_video_grid(
    video_path,
    output_path,
    baseline_frames=30,
    difference_threshold=20,
    frame_skip=30,
    top_k=1000,
    eps=5,
    min_samples=2,
    max_clusters=1000,   # maximum clusters to keep per cell
    iou_threshold=0.5,
    visualize=True,
    cluster_circle_diff_min=40.0  
):
    """
    Process the video by first cropping it using an adaptive ROI and then dividing the ROI
    into a 3x3 grid (nine sections). For each grid cell:
      1. Build a baseline from the first `baseline_frames` frames.
      2. For subsequent frames (skipping as specified), compute the difference between the cell and its baseline.
      3. Apply a threshold and DBSCAN clustering.
      4. Before drawing each detected cluster, examine a 30×30 circular region (radius=15) 
         centered at the detected cluster’s center. Compute the average BGR value in that circle 
         (excluding the pixels already detected as part of the cluster). If the average is higher 
         than (100, 100, 100), remove the candidate.
      5. Draw the remaining clusters back on the full frame (using the ROI offset).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get original video dimensions.
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original video resolution: {orig_width}x{orig_height}, FPS={fps}, Total Frames={total_frames}")
    print(f"Skipping frames in steps of {frame_skip}")

    # --- Crop the video using the adaptive ROI ---
    y_roi, x_roi = calculate_adaptive_roi(orig_width, orig_height)
    rx, ry = x_roi[0], y_roi[0]
    rw = x_roi[1] - x_roi[0]
    rh = y_roi[1] - y_roi[0]
    print(f"Adaptive ROI: x from {rx} to {rx+rw}, y from {ry} to {ry+rh}")

    # For processing we use the ROI dimensions.
    proc_width = rw
    proc_height = rh

    # --- Define the grid within the ROI (3 columns x 3 rows) ---
    grid_rows = 13
    grid_cols = 13
    cell_w = proc_width // grid_cols
    cell_h = proc_height // grid_rows

    # Prepare dictionary to hold baseline data for each grid cell.
    # Keys are (row, col) tuples.
    baseline_data = {(r, c): [] for r in range(grid_rows) for c in range(grid_cols)}

    # ---------------- 1) Build baseline for each grid cell using the ROI ----------------
    frame_idx = 0
    while frame_idx < baseline_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
        # Crop the ROI from the full frame.
        roi = gray[ry:ry+rh, rx:rx+rw]
        # For each grid cell in the ROI.
        for r in range(grid_rows):
            for c in range(grid_cols):
                x_cell = c * cell_w
                y_cell = r * cell_h
                cell = roi[y_cell:min(y_cell+cell_h, proc_height), x_cell:min(x_cell+cell_w, proc_width)]
                baseline_data[(r, c)].append(cell)
        frame_idx += 1

    # Compute the mean baseline image for each grid cell.
    baseline_grid = {}
    for key, frames_list in baseline_data.items():
        if len(frames_list) == 0:
            raise ValueError(f"No baseline frames captured for grid cell {key}")
        baseline_grid[key] = np.mean(frames_list, axis=0).astype("float32")
    print("Built baseline for each grid cell.")

    # ---------------- 2) Prepare output video writer (if visualizing) ----------------
    if visualize:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Output the full original frame size.
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height), True)

    # ---------------- 3) Process frames (applying the process to each grid cell in the ROI) ----------------
    current_frame_index = baseline_frames
    processed_frames = 0

    while current_frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
        # Crop the ROI from the full frame.
        roi = gray[ry:ry+rh, rx:rx+rw]

        # Process each grid cell (cell coordinates are relative to the ROI)
        for r in range(grid_rows):
            for c in range(grid_cols):
                x_cell = c * cell_w
                y_cell = r * cell_h
                roi_current = roi[y_cell:min(y_cell+cell_h, proc_height), x_cell:min(x_cell+cell_w, proc_width)]
                base = baseline_grid[(r, c)]
                # Ensure the baseline image has the same size as the current cell.
                h_cell, w_cell = roi_current.shape
                if base.shape != roi_current.shape:
                    base = cv2.resize(base, (w_cell, h_cell))
                diff = cv2.absdiff(roi_current, base)
                diff_flat = diff.flatten()

                # Adative or Static threshold, choose one of them.

                mask_above = (diff_flat >= difference_threshold) # Static

                # Adaptive
                # adaptive_thresh = np.percentile(diff_flat, 95)
                # effective_threshold = max(adaptive_thresh, difference_threshold)
                # mask_above = (diff_flat >= effective_threshold)

                if not np.any(mask_above):
                    continue

                above_indices = np.where(mask_above)[0]
                if len(above_indices) > top_k:
                    partial_idxs = np.argpartition(diff_flat[above_indices], -top_k)[-top_k:]
                    top_k_in_above = above_indices[partial_idxs]
                else:
                    top_k_in_above = above_indices

                vals_topk = diff_flat[top_k_in_above]
                sort_desc = np.argsort(vals_topk)[::-1]
                top_k_in_above = top_k_in_above[sort_desc]

                coords = []
                values = []
                for idx_i in top_k_in_above:
                    row_i, col_i = divmod(idx_i, w_cell)
                    coords.append([col_i, row_i])
                    values.append(diff_flat[idx_i])
                if len(coords) == 0:
                    continue
                coords = np.array(coords, dtype=np.float32)
                values = np.array(values, dtype=np.float32)

                # Apply DBSCAN clustering on these cell-relative coordinates.
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
                labels = db.labels_
                unique_labels = set(labels) - {-1}

                raw_clusters = []
                for lb in unique_labels:
                    c_idx = np.where(labels == lb)[0]
                    c_vals = values[c_idx]
                    sum_diff = c_vals.sum()
                    raw_clusters.append({
                        'label': lb,
                        'indices': c_idx,
                        'sum_diff': sum_diff
                    })

                raw_clusters.sort(key=lambda c: c['sum_diff'], reverse=True)

                final_clusters = []
                for cl in raw_clusters:
                    c_indices = cl['indices']
                    c_coords = coords[c_indices]
                    c_vals = values[c_indices]

                    cluster_avg = c_vals.mean()
                    pts_for_circle = c_coords.reshape((-1, 1, 2))
                    (cx, cy), radius = cv2.minEnclosingCircle(pts_for_circle)
                    cx, cy, radius = float(cx), float(cy), float(radius)

                    # (Optional) Compute a bounding box for the cluster in the cell.
                    min_x_roi = max(int(cx - radius), 0)
                    max_x_roi = min(int(cx + radius + 1), w_cell)
                    min_y_roi = max(int(cy - radius), 0)
                    max_y_roi = min(int(cy + radius + 1), h_cell)

                    # (Optional) Build a set of detected cluster pixels (converted to full-frame coordinates).
                    cluster_pixel_set = set()
                    for idx_p in c_indices:
                        px, py = coords[idx_p]
                        gx = rx + x_cell + int(px)
                        gy = ry + y_cell + int(py)
                        cluster_pixel_set.add((gx, gy))

                    # Save cluster info.
                    cl.update({
                        'cluster_avg': cluster_avg,
                        'center': (cx, cy),
                        'radius': radius,
                        # Optionally, you could store a bounding box here as 'bbox'
                        # 'bbox': (min_x_roi, min_y_roi, max_x_roi, max_y_roi)
                    })
                    final_clusters.append(cl)

                final_clusters.sort(key=lambda c: c['sum_diff'], reverse=True)
                final_clusters = final_clusters[:max_clusters]

                # ---------------- Draw the detected clusters on the full frame ----------------
                color_palette = [
                    (0, 0, 255), (0, 255, 0), (255, 0, 0),
                    (255, 255, 0), (255, 0, 255), (0, 255, 255),
                    (128, 0, 128), (128, 128, 0), (0, 128, 128), (255, 128, 0)
                ]
                for i, fc in enumerate(final_clusters):
                    # Convert the cluster center to full-frame coordinates.
                    cx_int = rx + x_cell + int(fc['center'][0])
                    cy_int = ry + y_cell + int(fc['center'][1])
                    
                    # --- Candidate Check ---
                    # Define a 30x30 circle (radius 15) centered at (cx_int, cy_int).
                    check_radius = 15
                    x_min_c = max(cx_int - check_radius, 0)
                    x_max_c = min(cx_int + check_radius, orig_width)
                    y_min_c = max(cy_int - check_radius, 0)
                    y_max_c = min(cy_int + check_radius, orig_height)
                    
                    candidate_pixels = []
                    # For each pixel in the bounding box, check if it lies inside the circle.
                    # for yy in range(y_min_c, y_max_c):
                    #     for xx in range(x_min_c, x_max_c):
                    #         if (xx - cx_int)**2 + (yy - cy_int)**2 <= check_radius**2:
                    #             # Exclude pixels that are part of the detected cluster.
                    #             if (xx, yy) not in cluster_pixel_set:
                    #                 candidate_pixels.append(frame[yy, xx])
                    
                    if candidate_pixels:
                        avg_color = np.mean(candidate_pixels, axis=0)  # BGR order
                        # If every channel is higher than 100, remove candidate (skip drawing it)
                        if all(avg_color > 100):
                            continue  # Skip this cluster

                    # --- End Candidate Check ---

                    # Draw detected cluster pixels.
                    color = color_palette[i % len(color_palette)]
                    for idx_p in fc['indices']:
                        px, py = coords[idx_p]
                        gx = rx + x_cell + int(px)
                        gy = ry + y_cell + int(py)
                        cv2.circle(frame, (gx, gy), 3, color, 1)
                    
                    # Draw cluster center and enclosing circle.
                    cv2.circle(frame, (cx_int, cy_int), int(fc['radius']), color, 2)
                    info_text = f"diff={fc['sum_diff']:.1f}"
                    cv2.putText(frame, info_text, (cx_int, cy_int - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if visualize:
            out_writer.write(frame)

        current_frame_index += frame_skip

    cap.release()
    if visualize:
        out_writer.release()

    print(f"Done: Processed {processed_frames} frames (of {total_frames}), skip={frame_skip}.")
    if visualize:
        print(f"Output saved to {output_path}") 

if __name__ == "__main__":
    video_file = "/Users/kbh/Desktop/CyPhy/videos/IMG_3573.MOV"    # Path to your input video
    output_file = "result/glossy_2.mp4"

    process_video_grid(
        video_path=video_file,
        output_path=output_file,
        baseline_frames=5,
        difference_threshold=20,
        frame_skip=2,
        top_k=5,
        eps=5,
        min_samples=2,
        max_clusters=100,  # maximum clusters per grid cell
        visualize=True,
        cluster_circle_diff_min=10.0  
    )

