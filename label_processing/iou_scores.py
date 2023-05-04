


#Import Librairies
import pandas as pd
import torch
import numpy as np
import plotly.express as px
import plotly.io as pio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_iou(df_pred, df_gt):
    #Extract bounding boxes coordinates
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = df_pred
    class_gt, xmin_gt, ymin_gt, xmax_gt, ymax_gt = df_gt
    # Get the coordinates of the intersection rectangle
    x0_I = max(xmin_pred, xmin_gt)
    y0_I = max(ymin_pred, ymin_gt)
    x1_I = min(xmax_pred, xmax_gt)
    y1_I = min(ymax_pred, ymax_gt) 
    #Calculate width and height of the intersection area
    width_I = x1_I - x0_I 
    height_I = y1_I - y0_I  
    # Handle the negative value width or height of the intersection area
    if width_I<0 : width_I=0
    if height_I<0 : height_I=0
    # Calculate the intersection area:
    intersection = width_I * height_I   
    # Calculate the union area:
    width_A, height_A = xmax_pred - xmin_pred, ymax_pred - ymin_pred
    width_B, height_B = xmax_gt - xmin_gt, ymax_gt - ymin_gt
    union = (width_A * height_A) + (width_B * height_B) - intersection   
    # Calculate the IoU:
    IoU = intersection/union   
    # for plotting purpose 
    boxI = torch.tensor([x0_I, y0_I, width_I,height_I])    
    # Return the IoU and intersection box
    return IoU 

def comparison(df_pred, df_gt):
    max_scores = []
    max_coords = []
    for _, row_pred in df_pred.iterrows():
        max_score: float = 0
        pred_coords = (row_pred.xmin_pred, row_pred.ymin_pred, row_pred.xmax_pred, row_pred.ymax_pred)
        for _, row_gt in df_gt.iterrows():
            gt_coords = (row_gt.class_gt, row_gt.xmin_gt, row_gt.ymin_gt, row_gt.xmax_gt, row_gt.ymax_gt)
            iou = calculate_iou(pred_coords, gt_coords)
            if iou > max_score:
                max_score = iou
                max_coord = gt_coords
        max_coords.append(max_coord)
        max_scores.append(max_score)
    df_pred.score = max_scores
    df_pred[["class_gt","xmin_gt", "ymin_gt", "xmax_gt", "ymax_gt"]] = max_coords
    return df_pred


def concat_frames(df_pred, df_gt, folder):
    df_gt = pd.read_csv(df_gt)
    df_pred = pd.read_csv(df_pred)
    df_gt.rename(columns={"class": "class_gt", "xmin": "xmin_gt", "ymin": "ymin_gt", "xmax": "xmax_gt", "ymax": "ymax_gt"}, inplace = True)
    df_pred.rename(columns={"class": "class_pred", "xmin": "xmin_pred", "ymin": "ymin_pred", "xmax": "xmax_pred", "ymax": "ymax_pred"}, inplace = True)
    frames = []
    for element in df_pred.filename.unique():
        df_pred_filename = df_pred[df_pred.filename == element]
        df_gt_filename = df_gt[df_gt.filename == element]
        new_frame = comparison(df_pred_filename, df_gt_filename)
        frames.append(new_frame)
    df = pd.concat(frames)
    filepath = Path(f'{folder}/iou_scores.csv')
    df.to_csv(filepath)
    return df


def box_plot_iou(df_pred, df_gt, folder):
    IOU_df = concat_frames(df_pred, df_gt, folder)
    fig = px.box(IOU_df, y="score", points="all", color="class_pred")
    fig.update_layout(title_text="IOU Scores")
    pio.write_image(fig, f'{folder}/iou_box.pdf')

def class_pred(df_pred, df_gt, folder):
    IOU_df = concat_frames(df_pred, df_gt, folder)
    conditions = [(IOU_df['class_pred'] == IOU_df['class_gt']), #define conditions
                 (IOU_df['class_pred'] != IOU_df['class_gt'])]
    choices = ['match', 'no_match'] #define choices
    IOU_df['comparison_values'] = np.select(conditions, choices, default='Tie') #create new column in DataFrame that displays results of comparisons
    # Plot
    colors = ['steelblue', 'firebrick']
    fig = px.histogram(IOU_df, x='class_pred', color="comparison_values").update_xaxes(categoryorder='total descending')
    fig.update_layout(xaxis_type='category', title_text='match VS no_match')
    pio.write_image(fig, f'{folder}/class_pred.pdf')
    
def accuracy_segmentation(df_pred, df_gt, folder):
    concat_frames(df_pred, df_gt, folder)
    box_plot_iou(df_pred, df_gt, folder)
    class_pred(df_pred, df_gt, folder)
    
